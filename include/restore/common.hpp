#ifndef RESTORE_COMMON_H
#define RESTORE_COMMON_H

#include <cstdint>

namespace ReStore {

// Defines how the serialized blocks are aligned in memory.
// See the documentation for offsetMode() for details.
enum class OffsetMode : uint8_t { constant, lookUpTable };

// Global and local id. The global id is unique across all ranks, that is each copy of the
// same block has the same global id on all ranks. We can use the global id to request a
// block.
// TODO Do we need local ids? If we do, describe the difference between global and local block ids
using block_id_t = std::size_t;

} // End of namespace ReStore

#endif // Include guard