#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <optional>
#include <vector>

class ReStoreMPIContext {
    public:
    ReStoreMPIContext(MPI_Comm comm) : _comm(comm) {}

    void updateComm(MPI_Comm comm) { _comm = comm; }

    struct OriginalRank {
        int rank;
            operator int() const { return rank; }
    };

    struct CurrentRank {
        int rank;
            operator int() const { return rank; }
    };

    struct Message {
        std::shared_ptr<uint8_t> data;
        int                      size;
        CurrentRank              rank;
    };

    OriginalRank getOriginalRank(CurrentRank rank) { return {rank.rank}; }

    std::optional<CurrentRank> getCurrentRank(OriginalRank rank) { return {}; }

    bool isAlive(OriginalRank rank) { return false; }


    std::vector<Message> SparseAllToAll(const std::vector<Message>& messages) { return {}; }

    private:
    MPI_Comm _comm;
};

#endif // MPI_CONTEXT_H
