# ReStore: In-Memory REplicated STORagE for Rapid Recovery in Fault-Tolerant Algorithms

Fault-tolerant distributed applications require mechanisms to recover data lost via a process failure.
On modern cluster systems it is typically impractical to request replacement resources after such a failure.
Therefore, applications have to continue working with the remaining resources.
This requires redistributing the workload and that the non-failed processes reload the lost data.
ReStore is a C++ header-only library for MPI programs that enables recovery of lost data after (a) process failure(s).
By storing all required data in memory via an appropriate data distribution and replication, recovery is substantially faster than with standard checkpointing schemes that rely on a parallel file system.
As you as the application programmer can specify which data to load ReStore also supports shrinking recovery instead of recovery using spare compute nodes.

## Including ReStore into your application

To use ReStore, first add the repository as a submodule into your project:
```Bash
git submodule add --recursive https://github.com/ReStoreCpp/ReStore.git extern/ReStore
```

Then, include the following into your CMakeLists.txt:
```CMake
# Configure and link ReStore
set(ReStore_BUILD_TESTS Off)
set(ReStore_BUILD_BENCHMARKS Off)
set(ReStore_ID_RANDOMIZATION On)

add_subdirectory(extern/ReStore)
target_link_libraries(${YOUR_TARGETS_NAME} ReStore)
```

You can use ID-randomization to break up access patterns in your `load` requests.
If enabled, the block IDs you provide will be permuted using a pseudorandom-projection.
If you then for example access a range of consecutive blocks IDs, e.g. after the PE which worked on these IDs failed; more PEs will be able to serve the request, resulting in a speedup.
If you request most or all of the data `submitted` in each `load`, turning ID-randomization of will be faster.
See Hespe and Hübner et al. (2022) [1] for details.

## Code examples

### The general use case

This example shows the general usage of ReStore.

```cpp
#include <core.hpp>

// First, create the restore object.
ReStore::ReStore<YourAwesomeDatatype> store(
    MPI_COMM_WORLD, // MPI communicator to use. ULFM currently supports only MPI_COMM_WORLD.
    4,              // Replication level, 3 or 4 are sane defaults.
    ReStore::OffsetMode::constant, // Currently, the only supported mode.
    sizeof(YourAwesomeDatatype)    // Your block size, use at least 64 bytes.
);

// Next, submit you data to the ReStore, if a failure happened between creation of the ReStore
// and the submission of the data, please re-create the ReStore.
ReStore::block_id_t blockId = 0;
store.submitBlocks(
    // The serialization function; your can stream your data to the provided stream using
    // the << operator.
    [](const YourAwesomeDatatype& value, ReStore::SerializedBlockStoreStream& stream) {
        // Either use:
        stream << value;
        // or, for big, already consecutively stored data:
        stream.writeBytes(constBytePtr, sizeof(YourAwesomeDatatype));
        },
    // The enumerator function; should return nullopt if there are no more blocks to submit
    // on this PE.
    [localBlockId, ...]() {
        auto ret = numberOfBlocksOnThisPE == localBlockId
                        ? std::nullopt
                        : std::make_optional(ReStore::NextBlock<YourAwesomeDatatype>(
                            {globalBlockId(localBlockId), constRefToYourDataForThisBlock}));
        blockId++;  // We cannot put this in the above line, as we can't assume if the first
                    // argument of the pair is bound before or after the increment.
        return ret;
    },
    globalNumberOfBlocks
);

// A failure occurred; set ReStore's communicator to the fixed communicator obtained by
// MPIX_Comm_shrink()
store.updateComm(newComm);

// Next, request the data you need on each PE.
// requestedBlocks is of type
// std::vector<std::pair<ReStore::block_id_t, size_t>>
// [ (firstBlockIdOfRange1, numberOfBlocks1), (firstBlockIdOfRange2, numberOfBlocks2), ...]
store.pullBlocks(
    requestedBlocks,
    //  De-serialization function.
    [...] (const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
        // ...
});

```
### Data stored in a std::vector

If your data resides in a `std::vector`, you can use the ReStore-provided wrapper.

```cpp
#include <restore/core.hpp>
#include <restore/restore_vector.hpp>

// Create the ReStoreVector wrapper.
ReStore::ReStoreVector<YourAwesomeDatatype>> reStoreVectorWrapper(
    blockSizeInBytes, // Can for example be used to group all dimensions of a single data point.
    MPI_COMM_WORLD,
    replicationLevel,
    blocksPerPermutationRange, // defaults to 4096
    paddingValue, // The value used to pad the data; defaults to 0
);

// Submit your data to the ReStore.
const auto numBlocksLocal = reStoreVectorWrapper->submitData(referenceToYourDataVector);

// After a failure
reStoreVectorWrapper.updateComm(newComm); // see above

reStoreVectorWrapper.restoreDataAppendPullBlocks(
    referenceToVectorContainingYourData, // ReStore will append the new data points at the end.
    requestedBlocks, // see above
);
```

### A simple load-balancer

You can use the ReStore-provided LoadBalancer.
If a PE fails, it will help you with calculating the new distribution of blocks to PEs.
Each surviving PE will get an equal share of the blocks residing on each PE that failed.
This of course works for multiple rounds of failing PEs, too.

```cpp
#include <restore/core.hpp>
#include <restore/equal_load_balancer.hpp>

// Describes, which block range (firstBlockId, numberOfBlocks) resides on which PE.
using BlockRange     = std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::original_rank_t>;
using BlockRangeList = std::vector<BlockRange>;

// Create the LoadBalancer object.
ReStore::EqualLoadBalancer loadBalancer(blockRangeList, numberOfPEs)

// After a failure, let the LoadBalancer decide which PE gets which data points:
const auto newBlocks = _loadBalancer.getNewBlocksAfterFailureForPullBlocks(
    ranksDiedSinceLastCall, myRankWhenCreatingTheLoadBalancer
);
// You can hand newBlocks to restore.pullBlocks() or the ReStoreVector wrapper.

// If everyone completed the restoration successfully, we can commit to the new data distribution. If
// there was another PE failure in the meantime, you can re-call getNewBlocksAfterFailureForPullBlocks.
_loadBalancer.commitToPreviousCall();

// Further failures, repeat the above steps.
```

## Publication
If you use ReStore in your research, please cite the following paper:

[1] TODO

