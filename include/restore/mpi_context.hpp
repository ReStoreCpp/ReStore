class ReStoreMPIContext {
public:
    ReStoreMPIContext(MPI_Comm comm) : _comm(comm);

    void updateComm(MPI_Comm comm);

    template<typename T>
    struct Message {
        std::shared_pointer<T> data;
        int size;
        int rank;
    };

    template<typename T>
    std::vector<Message<T>> SparseAllToall(const std::vector<Message<T>> &messages);

private:
    MPI_Comm _comm;
}