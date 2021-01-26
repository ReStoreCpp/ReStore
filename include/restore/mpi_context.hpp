#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <optional>
#include <vector>

namespace ReStoreMPI {

enum class OriginalRank : int {};
enum class CurrentRank : int {};

struct Message {
    std::shared_ptr<uint8_t> data;
    int                      size;
    CurrentRank              rank;
};

class MPIContext {
    public:
    MPIContext(MPI_Comm comm) : _comm(comm) {}

    void updateComm(MPI_Comm comm) { _comm = comm; }

    OriginalRank getOriginalRank(CurrentRank rank) { return static_cast<OriginalRank>(rank); }

    std::optional<CurrentRank> getCurrentRank(OriginalRank rank) { return {}; }

    bool isAlive(OriginalRank rank) { return false; }

    std::vector<Message> SparseAllToAll(const std::vector<Message>& messages) { return {}; }

    private:
    MPI_Comm _comm;
};

} // namespace ReStoreMPI

#endif // MPI_CONTEXT_H
