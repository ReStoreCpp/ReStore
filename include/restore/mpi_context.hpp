#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include <cassert>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <optional>
#include <vector>

#ifndef RESTORE_SPARSE_ALL_TO_ALL_TAG
    #define RESTORE_SPARSE_ALL_TO_ALL_TAG 42
#endif

namespace ReStoreMPI {

enum class OriginalRank : int {};
enum class CurrentRank : int {};

struct Message {
    std::shared_ptr<uint8_t> data;
    int                      size;
    CurrentRank              rank;
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

    OriginalRank getOriginalRank(const CurrentRank rank) const {
        int currentRank = static_cast<int>(rank);
        int originalRank;
        MPI_Group_translate_ranks(_currentGroup, 1, &currentRank, _originalGroup, &originalRank);
        assert(originalRank != MPI_UNDEFINED);
        return static_cast<OriginalRank>(originalRank);
    }

    std::optional<CurrentRank> getCurrentRank(const OriginalRank rank) const {
        int originalRank = static_cast<int>(rank);
        int currentRank;
        MPI_Group_translate_ranks(_originalGroup, 1, &originalRank, _currentGroup, &currentRank);
        return currentRank != MPI_UNDEFINED ? std::optional<CurrentRank>(static_cast<CurrentRank>(currentRank))
                                            : std::nullopt;
    }

    private:
    MPI_Group _originalGroup;
    MPI_Group _currentGroup;
};

void receiveNewMessage(std::vector<Message>& result, const MPI_Comm comm, const int tag) {
    int        newMessageReceived = false;
    MPI_Status receiveStatus;
    MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &newMessageReceived, &receiveStatus);
    if (newMessageReceived) {
        assert(receiveStatus.MPI_TAG == tag);
        int size;
        MPI_Get_count(&receiveStatus, MPI_BYTE, &size);
        result.emplace_back(Message{std::shared_ptr<uint8_t>(new uint8_t[size]), size,
                                    static_cast<CurrentRank>(receiveStatus.MPI_SOURCE)});
        MPI_Recv(
            result.back().data.get(), size, MPI_BYTE, receiveStatus.MPI_SOURCE, receiveStatus.MPI_TAG, comm,
            &receiveStatus);
    }
}

std::vector<Message> SparseAllToAll(const std::vector<Message>& messages, const MPI_Comm& comm, const int tag) {
    // Send all messages
    std::vector<MPI_Request> requests(messages.size());
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto&  message    = messages[i];
        MPI_Request* requestPtr = &requests[i];
        MPI_Issend(message.data.get(), message.size, MPI_BYTE, static_cast<int>(message.rank), tag, comm, requestPtr);
    }

    // Receive messages until all messages sent have been received
    int                  allSendsFinished = false;
    std::vector<Message> result;
    while (!allSendsFinished) {
        receiveNewMessage(result, comm, tag);
        // This might be improved by using the status and removing all finished requests
        MPI_Testall(requests.size(), requests.data(), &allSendsFinished, MPI_STATUSES_IGNORE);
    }

    // Enter a barrier. Once all PEs are here, we know that all messages have been received
    MPI_Request barrierRequest;
    MPI_Ibarrier(comm, &barrierRequest);

    // Continue receiving messages until the barrier completes
    // (and thus all messages from all PEs have been received)
    int barrierFinished = false;
    while (!barrierFinished) {
        receiveNewMessage(result, comm, tag);
        MPI_Status barrierStatus;
        MPI_Test(&barrierRequest, &barrierFinished, &barrierStatus);
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

    OriginalRank getOriginalRank(const CurrentRank rank) const {
        return _rankManager.getOriginalRank(rank);
    }

    std::optional<CurrentRank> getCurrentRank(const OriginalRank rank) const {
        return _rankManager.getCurrentRank(rank);
    }

    bool isAlive(const OriginalRank rank) const {
        return getCurrentRank(rank).has_value();
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
