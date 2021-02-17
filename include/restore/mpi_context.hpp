#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include <algorithm>
#include <bits/stdint-uintn.h>
#include <cassert>
#include <cstdint>
#include <exception>
#include <memory>
#include <mpi.h>
#include <optional>
#include <vector>

#ifndef RESTORE_SPARSE_ALL_TO_ALL_TAG
    #define RESTORE_SPARSE_ALL_TO_ALL_TAG 42
#endif

namespace ReStoreMPI {

typedef int current_rank_t;
typedef int original_rank_t;

struct SendMessage {
    const uint8_t* data;
    int            size;
    current_rank_t destRank;

    SendMessage(const uint8_t* _data, const int _size, const current_rank_t _destRank) noexcept
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
    std::vector<uint8_t> data;
    current_rank_t       srcRank;

    RecvMessage(const size_t size, const current_rank_t _srcRank) : data(size), srcRank(_srcRank) {}
    RecvMessage(std::vector<uint8_t>&& _data, const current_rank_t _srcRank)
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
    RankManager(MPI_Comm comm) {
        MPI_Comm_group(comm, &_originalGroup);
        MPI_Comm_group(comm, &_currentGroup);
    }

    void updateComm(MPI_Comm newComm) {
        MPI_Group_free(&_currentGroup);
        MPI_Comm_group(newComm, &_currentGroup);
    }

    void resetOriginalCommToCurrentComm() {
        MPI_Group_free(&_originalGroup);
        _originalGroup = _currentGroup;
    }

    original_rank_t getOriginalSize() {
        original_rank_t size;
        MPI_Group_size(_originalGroup, &size);
        return size;
    }

    original_rank_t getMyOriginalRank() {
        original_rank_t rank;
        MPI_Group_rank(_originalGroup, &rank);
        return rank;
    }

    current_rank_t getCurrentSize() {
        current_rank_t size;
        MPI_Group_size(_currentGroup, &size);
        return size;
    }

    current_rank_t getMyCurrentRank() {
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
        std::vector<current_rank_t> out(in.size());
        MPI_Group_translate_ranks(_originalGroup, (int)in.size(), in.data(), _currentGroup, out.data());
        for (size_t i = 0; i < in.size(); ++i) {
            out[i] = out[i] == MPI_UNDEFINED ? MPI_UNDEFINED : in[i];
        }
        out.erase(
            std::remove_if(out.begin(), out.end(), [](const current_rank_t& rank) { return rank == MPI_UNDEFINED; }),
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

    private:
    MPI_Group _originalGroup;
    MPI_Group _currentGroup;
};

template <class F>
void successOrThrowMpiCall(const F& mpiCall) {
    int rc, ec;
    rc = mpiCall();
    MPI_Error_class(rc, &ec);
    if (ec == MPI_ERR_PROC_FAILED || ec == MPI_ERR_PROC_FAILED_PENDING) {
        throw FaultException();
    }
    if (ec == MPI_ERR_REVOKED) {
        throw RevokedException();
    }
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
    MPIContext(MPI_Comm comm) : _comm(comm), _rankManager(comm) {}

    void updateComm(MPI_Comm newComm) {
        _comm = newComm;
        _rankManager.updateComm(newComm);
    }

    void resetOriginalCommToCurrentComm() {
        _rankManager.resetOriginalCommToCurrentComm();
    }

    original_rank_t getOriginalSize() {
        return _rankManager.getOriginalSize();
    }

    original_rank_t getMyOriginalRank() {
        return _rankManager.getMyOriginalRank();
    }

    current_rank_t getCurrentSize() {
        return _rankManager.getCurrentSize();
    }

    current_rank_t getMyCurrentRank() {
        return _rankManager.getMyCurrentRank();
    }

    original_rank_t getOriginalRank(const current_rank_t rank) const {
        return _rankManager.getOriginalRank(rank);
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

    private:
    MPI_Comm    _comm;
    RankManager _rankManager;
};

} // namespace ReStoreMPI

#endif // MPI_CONTEXT_H
