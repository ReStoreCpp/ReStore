#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <stdint.h>
#include <vector>

#include "restore/helpers.hpp"

#if USE_FTMPI
    #include <mpi-ext.h>
#endif

#ifndef RESTORE_SPARSE_ALL_TO_ALL_TAG
    #define RESTORE_SPARSE_ALL_TO_ALL_TAG 42
#endif

namespace ReStoreMPI {

typedef int current_rank_t;
typedef int original_rank_t;

struct SendMessage {
    static_assert(
        sizeof(std::byte) == sizeof(char),
        "byte and char have different sizes. This means restore will probably not work and show undefined behavior.");
    const std::byte* data;
    int              size;
    current_rank_t   destRank;

    SendMessage(const std::byte* _data, const int _size, const current_rank_t _destRank) noexcept
        : data(_data),
          size(_size),
          destRank(_destRank) {}

    // Performs a deep comparison, i.e. the contents of the message is checked for equality, not where data points to.
    bool operator==(const SendMessage& that) const noexcept {
        assert(this->data);
        assert(that.data);
        assert(this->size >= 0);
        assert(that.size >= 0);

        if (this->size != that.size || this->destRank != that.destRank) {
            return false;
        } else {
            assert(this->size == that.size);
            for (decltype(this->size) idx = 0; idx < this->size; ++idx) {
                if (this->data[idx] != that.data[idx]) {
                    return false;
                }
            }
            return true;
        }
    }

    // Performs a deep comparison, i.e. the contents of the message is checked for inequality, not where data points to.
    bool operator!=(const SendMessage& that) const noexcept {
        return !(*this == that);
    }
};

struct RecvMessage {
    std::vector<std::byte> data;
    current_rank_t         srcRank;

    RecvMessage(const size_t size, const current_rank_t _srcRank) : data(size), srcRank(_srcRank) {}
    RecvMessage(std::vector<std::byte>&& _data, const current_rank_t _srcRank) noexcept
        : data(std::move(_data)),
          srcRank(_srcRank) {}

    // Performs a deep comparison, i.e. the contents of the message is checked for equality, not where data points to.
    bool operator==(const RecvMessage& that) const {
        return this->srcRank == that.srcRank && this->data == that.data;
    }

    // Performs a deep comparison, i.e. the contents of the message is checked for inequality, not where data points to.
    bool operator!=(const RecvMessage& that) const {
        return !(*this == that);
    }
};

// Just for debugging
// std::ostream& operator<<(std::ostream& out, const ReStoreMPI::RecvMessage& v) {
//     out << "RecvMessage(";
//     out << v.data.size();
//     out << ", " << v.srcRank;
//     out << ")";
//     return out;
// }

class FaultException : public std::exception {
    virtual const char* what() const throw() override {
        return "A rank in the communicator failed";
    }
};

class RevokedException : public std::exception {
    virtual const char* what() const throw() override {
        return "The communicator used has been revoked. Call updateComm with the new communicator before trying to "
               "communicate again.";
    }
};

class RankManager {
    public:
    explicit RankManager(MPI_Comm comm) {
        MPI_Comm_group(comm, &_originalGroup);
        MPI_Comm_group(comm, &_currentGroup);
        MPI_Comm_group(comm, &_lastDiedRanksRequestedGroup);
    }

    void updateComm(MPI_Comm newComm) {
        MPI_Group_free(&_currentGroup);
        MPI_Comm_group(newComm, &_currentGroup);
    }

    void resetOriginalCommToCurrentComm() {
        MPI_Group_free(&_originalGroup);
        _originalGroup = _currentGroup;
    }

    original_rank_t getOriginalSize() const {
        original_rank_t size;
        MPI_Group_size(_originalGroup, &size);
        return size;
    }

    original_rank_t getMyOriginalRank() const {
        original_rank_t rank;
        MPI_Group_rank(_originalGroup, &rank);
        return rank;
    }

    current_rank_t getCurrentSize() const {
        current_rank_t size;
        MPI_Group_size(_currentGroup, &size);
        return size;
    }

    current_rank_t getMyCurrentRank() const {
        current_rank_t rank;
        MPI_Group_rank(_currentGroup, &rank);
        return rank;
    }

    original_rank_t getOriginalRank(const current_rank_t currentRank) const {
        int originalRank;
        MPI_Group_translate_ranks(_currentGroup, 1, &currentRank, _originalGroup, &originalRank);
        assert(originalRank != MPI_UNDEFINED);
        return originalRank;
    }

    std::optional<current_rank_t> getCurrentRank(const original_rank_t originalRank) const {
        int currentRank;
        MPI_Group_translate_ranks(_originalGroup, 1, &originalRank, _currentGroup, &currentRank);
        return currentRank != MPI_UNDEFINED ? std::optional<current_rank_t>(currentRank) : std::nullopt;
    }

    std::vector<original_rank_t> getOnlyAlive(const std::vector<original_rank_t>& in) const {
        std::vector<original_rank_t> out(in.size());
        MPI_Group_translate_ranks(_originalGroup, (int)in.size(), in.data(), _currentGroup, out.data());
        for (size_t i = 0; i < in.size(); ++i) {
            out[i] = out[i] == MPI_UNDEFINED ? MPI_UNDEFINED : in[i];
        }
        out.erase(
            std::remove_if(out.begin(), out.end(), [](const original_rank_t& rank) { return rank == MPI_UNDEFINED; }),
            out.end());
        return out;
    }

    std::vector<current_rank_t> getAliveCurrentRanks(const std::vector<original_rank_t>& originalRanks) const {
        std::vector<current_rank_t> currentRanks(originalRanks.size());
        MPI_Group_translate_ranks(
            _originalGroup, (int)originalRanks.size(), originalRanks.data(), _currentGroup, currentRanks.data());
        currentRanks.erase(
            std::remove_if(
                currentRanks.begin(), currentRanks.end(),
                [](const current_rank_t& rank) { return rank == MPI_UNDEFINED; }),
            currentRanks.end());
        return currentRanks;
    }

    std::vector<original_rank_t> getRanksDiedSinceLastCall() {
        MPI_Group difference;
        MPI_Group_difference(_lastDiedRanksRequestedGroup, _currentGroup, &difference);
        int numRanksDied;
        MPI_Group_size(difference, &numRanksDied);
        std::vector<int> groupRankIds(static_cast<size_t>(numRanksDied));
        std::iota(std::begin(groupRankIds), std::end(groupRankIds), 0);
        std::vector<int> originalRankIds(static_cast<size_t>(numRanksDied));
        MPI_Group_translate_ranks(
            difference, numRanksDied, groupRankIds.data(), _originalGroup, originalRankIds.data());
        MPI_Group_free(&_lastDiedRanksRequestedGroup);
        _lastDiedRanksRequestedGroup = _currentGroup;
        return originalRankIds;
    }

    private:
    MPI_Group _originalGroup;
    MPI_Group _currentGroup;
    MPI_Group _lastDiedRanksRequestedGroup;
};

template <class F>
void successOrThrowMpiCall(const F& mpiCall) {
#if USE_FTMPI
    int rc, ec;
    rc = mpiCall();
    MPI_Error_class(rc, &ec);
    if (ec == MPI_ERR_PROC_FAILED || ec == MPI_ERR_PROC_FAILED_PENDING) {
        throw FaultException();
    }
    if (ec == MPI_ERR_REVOKED) {
        throw RevokedException();
    }
#else
    mpiCall();
#endif
}

void receiveNewMessage(std::vector<RecvMessage>& result, const MPI_Comm comm, const int tag) {
    int        newMessageReceived = false;
    MPI_Status receiveStatus;
    successOrThrowMpiCall([&]() { return MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &newMessageReceived, &receiveStatus); });
    if (newMessageReceived) {
        assert(receiveStatus.MPI_TAG == tag);
        int size;
        MPI_Get_count(&receiveStatus, MPI_BYTE, &size);
        result.emplace_back(size, receiveStatus.MPI_SOURCE);
        successOrThrowMpiCall([&]() {
            return MPI_Recv(
                result.back().data.data(), size, MPI_BYTE, receiveStatus.MPI_SOURCE, receiveStatus.MPI_TAG, comm,
                &receiveStatus);
        });
    }
}

std::vector<RecvMessage> SparseAllToAll(const std::vector<SendMessage>& messages, const MPI_Comm& comm, const int tag) {
    // Send all messages
    std::vector<MPI_Request> requests(messages.size());
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto&  message    = messages[i];
        MPI_Request* requestPtr = &requests[i];
        successOrThrowMpiCall([&]() {
            return MPI_Issend(message.data, message.size, MPI_BYTE, message.destRank, tag, comm, requestPtr);
        });
    }

    // Receive messages until all messages sent have been received
    int                      allSendsFinished = false;
    std::vector<RecvMessage> result;
    while (!allSendsFinished) {
        receiveNewMessage(result, comm, tag);
        // This might be improved by using the status and removing all finished requests
        successOrThrowMpiCall([&]() {
            return MPI_Testall((int)requests.size(), requests.data(), &allSendsFinished, MPI_STATUSES_IGNORE);
        });
    }

    // Enter a barrier. Once all PEs are here, we know that all messages have been received
    MPI_Request barrierRequest;
    successOrThrowMpiCall([&]() { return MPI_Ibarrier(comm, &barrierRequest); });

    // Continue receiving messages until the barrier completes
    // (and thus all messages from all PEs have been received)
    int barrierFinished = false;
    while (!barrierFinished) {
        receiveNewMessage(result, comm, tag);
        MPI_Status barrierStatus;
        successOrThrowMpiCall([&]() { return MPI_Test(&barrierRequest, &barrierFinished, &barrierStatus); });
    }
    return result;
}


class MPIContext {
    public:
    explicit MPIContext(MPI_Comm comm) : _comm(comm), _rankManager(comm) {}

    void updateComm(MPI_Comm newComm) {
        _comm = newComm;
        _rankManager.updateComm(newComm);
    }

    MPI_Comm getComm() const {
        return _comm;
    }

    void resetOriginalCommToCurrentComm() {
        _rankManager.resetOriginalCommToCurrentComm();
    }

    original_rank_t getOriginalSize() const {
        return _rankManager.getOriginalSize();
    }

    original_rank_t getMyOriginalRank() const {
        return _rankManager.getMyOriginalRank();
    }

    current_rank_t getCurrentSize() const {
        return _rankManager.getCurrentSize();
    }

    current_rank_t getMyCurrentRank() const {
        return _rankManager.getMyCurrentRank();
    }

    original_rank_t getOriginalRank(const current_rank_t rank) const {
        return _rankManager.getOriginalRank(rank);
    }

    original_rank_t numFailuresSinceReset() const {
        return getOriginalSize() - getCurrentSize();
    }

    std::vector<original_rank_t> getRanksDiedSinceLastCall() {
        return _rankManager.getRanksDiedSinceLastCall();
    }

    std::optional<current_rank_t> getCurrentRank(const original_rank_t rank) const {
        return _rankManager.getCurrentRank(rank);
    }

    bool isAlive(const original_rank_t rank) const {
        return getCurrentRank(rank).has_value();
    }

    std::vector<original_rank_t> getOnlyAlive(const std::vector<original_rank_t>& in) const {
        return _rankManager.getOnlyAlive(in);
    }

    std::vector<current_rank_t> getAliveCurrentRanks(const std::vector<original_rank_t>& originalRanks) const {
        return _rankManager.getAliveCurrentRanks(originalRanks);
    }

    std::vector<RecvMessage>
    SparseAllToAll(const std::vector<SendMessage>& messages, const int tag = RESTORE_SPARSE_ALL_TO_ALL_TAG) const {
        return ReStoreMPI::SparseAllToAll(messages, _comm, tag);
    }

    // TODO compile time enable exceptions instead of assertions
    template <class data_t>
    void broadcast(data_t* data, size_t numDataElements = 1, int root = 0) {
        static_assert(std::is_pod_v<data_t>, "broadcast only works for POD data");
        int _numDataElements = asserting_cast<int>(numDataElements);
        return successOrThrowMpiCall(
            [&]() { return MPI_Bcast(data, _numDataElements, get_mpi_type<data_t>(), root, _comm); });
    }

    template <class data_t>
    std::vector<data_t> broadcast(std::vector<data_t>& data, int root = 0) {
        static_assert(std::is_pod_v<data_t>, "broadcast only works for POD data");
        return broadcast(data.data(), data.size(), root, _comm);
    }

    template <class data_t>
    void allreduce(data_t* data, MPI_Op operation, size_t numDataElements = 1) {
        static_assert(std::is_pod_v<data_t>, "allreduce only works for POD data");
        int _numDataElements = asserting_cast<int>(numDataElements);
        return successOrThrowMpiCall([&]() {
            return MPI_Allreduce(MPI_IN_PLACE, data, _numDataElements, get_mpi_type<data_t>(), operation, _comm);
        });
    }

    template <class data_t>
    void allreduce(std::vector<data_t>& data, MPI_Op operation) {
        static_assert(std::is_pod_v<data_t>, "allreduce only works for POD data");
        return allreduce(data.data(), operation, data.size());
    }

    template <class data_t>
    std::vector<data_t> gatherv(std::vector<data_t>& data, int root = 0) {
        static_assert(std::is_pod_v<data_t>, "gatherv only works for POD data");

        // First, gather the number of data elements per rank
        int myNumDataElements = throwing_cast<int>(data.size());

        std::vector<int> numDataElementsPerRank;
        if (_rankManager.getMyCurrentRank() == root) {
            numDataElementsPerRank.resize(asserting_cast<size_t>(_rankManager.getCurrentSize()));
        }

        successOrThrowMpiCall([&]() {
            return MPI_Gather(
                &myNumDataElements,                          // send buffer
                1,                                           // send count
                get_mpi_type<decltype(myNumDataElements)>(), // send type
                numDataElementsPerRank.data(),               // receive buffer
                1,                                           // receive count
                get_mpi_type<decltype(myNumDataElements)>(), // receive type
                root,                                        // root
                _comm                                        // communicator
            );
        });

        // Next, compute the displacements for the gatherv operation
        std::vector<int> displacements(asserting_cast<size_t>(_rankManager.getCurrentSize()) + 1, 0);
        assert(_rankManager.getMyCurrentRank() != root || numDataElementsPerRank.size() + 1 == displacements.size());

        std::partial_sum(numDataElementsPerRank.begin(), numDataElementsPerRank.end(), displacements.begin() + 1);
        assert(displacements[0] == 0);

        auto numDataElementsGlobal = asserting_cast<size_t>(displacements[displacements.size() - 1]);
        assert(_rankManager.getMyCurrentRank() != root || numDataElementsGlobal > myNumDataElements);
        assert(_rankManager.getMyCurrentRank() == root || numDataElementsGlobal == 0);
        assert(_rankManager.getMyCurrentRank() != root || numDataElementsGlobal > 0);

        // Finally, gatherv the data
        std::vector<data_t> receiveBuffer(numDataElementsGlobal, 0);
        assert(receiveBuffer.size() == numDataElementsGlobal);

        successOrThrowMpiCall([&]() {
            return MPI_Gatherv(
                data.data(),                            // send buffer
                asserting_cast<int>(myNumDataElements), // send count
                get_mpi_type<data_t>(),                 // send type
                receiveBuffer.data(),                   // receive buffer
                numDataElementsPerRank.data(),          // receive count
                displacements.data(),                   // displacements into the receive buffer
                get_mpi_type<data_t>(),                 // receive type
                0, MPI_COMM_WORLD                       // root rank and communicator
            );
        });

        return receiveBuffer;
    }

#ifdef USE_FTMPI
    void ft_barrier() {
        successOrThrowMpiCall([&]() {
            int flag = 42;
            return MPIX_Comm_agree(_comm, &flag);
        });
    }
#endif

    private:
    MPI_Comm    _comm;
    RankManager _rankManager;
};

} // namespace ReStoreMPI
#endif // MPI_CONTEXT_H
