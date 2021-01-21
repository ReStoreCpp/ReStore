#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include <memory>
#include <mpi/mpi.h>
#include <vector>

class ReStoreMPIContext {
    public:
    ReStoreMPIContext(MPI_Comm comm) : _comm(comm) {}

    void updateComm(MPI_Comm comm) { _comm = comm; }

    template <typename T>
    struct Message {
        std::shared_ptr<T> data;
        int                size;
        int                rank;
    };

    template <typename T>
    std::vector<Message<T>> SparseAllToall(const std::vector<Message<T>>& messages) {
        return {};
    }

    private:
    MPI_Comm _comm;
};

#endif // MPI_CONTEXT_H
