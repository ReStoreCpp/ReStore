#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include <algorithm>
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

struct Message {
    std::shared_ptr<uint8_t> data;
    int                      size;
    current_rank_t           rank;
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
        _originalGroup = _currentGroup;
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

void receiveNewMessage(std::vector<Message>& result, const MPI_Comm comm, const int tag) {
    int        newMessageReceived = false;
    MPI_Status receiveStatus;
    successOrThrowMpiCall([&]() { return MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &newMessageReceived, &receiveStatus); });
    if (newMessageReceived) {
        assert(receiveStatus.MPI_TAG == tag);
        int size;
        MPI_Get_count(&receiveStatus, MPI_BYTE, &size);
        result.emplace_back(
            Message{std::shared_ptr<uint8_t>(new uint8_t[(size_t)size]), size, receiveStatus.MPI_SOURCE});
        successOrThrowMpiCall([&]() {
            return MPI_Recv(
                result.back().data.get(), size, MPI_BYTE, receiveStatus.MPI_SOURCE, receiveStatus.MPI_TAG, comm,
                &receiveStatus);
        });
    }
}

std::vector<Message> SparseAllToAll(const std::vector<Message>& messages, const MPI_Comm& comm, const int tag) {
    // Send all messages
    std::vector<MPI_Request> requests(messages.size());
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto&  message    = messages[i];
        MPI_Request* requestPtr = &requests[i];
        successOrThrowMpiCall([&]() {
            return MPI_Issend(message.data.get(), message.size, MPI_BYTE, message.rank, tag, comm, requestPtr);
        });
    }

    // Receive messages until all messages sent have been received
    int                  allSendsFinished = false;
    std::vector<Message> result;
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

    std::vector<Message>
    SparseAllToAll(const std::vector<Message>& messages, const int tag = RESTORE_SPARSE_ALL_TO_ALL_TAG) const {
        return ReStoreMPI::SparseAllToAll(messages, _comm, tag);
    }

    private:
    MPI_Comm    _comm;
    RankManager _rankManager;
};

} // namespace ReStoreMPI

#endif // MPI_CONTEXT_H
