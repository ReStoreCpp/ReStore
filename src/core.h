#include <functional>
#include <utility>
#include <vector>
#include <optional>

template <class BlockType>
class Storage
{
    public:
        void submitBlocks(
                std::function<size_t(const BlockType&, void*)> serializeFunc,
                std::function<std::optional<std::pair<size_t globalId, BlockType&>>()> getNextBlock,
                bool canParallelize = false
        );

        /*
        pullBlocks(
                std::vector<std::pairs<size_t begin, size_t end>> blockRanges,
                std::function<void(void*, length, globalId)> handleSerializedBlock,
                bool canParallelize = false
        );
        */

        pushBlocks(
                std::vector<std::pair<std::pair<size_t begin, size_t end>, int dest>> blockRanges,
                std::function<void(void*, length, globalId)> handleSerializedBlock,
                bool canParallelize = false
        );
}

/*
Indended usage:

Storage storage;
// storage.setProcessMap(...) --- Skipped for now
storage.setReplication(k)
storage.setOffsetMode(constant|explicit, size_t c = 0)
storage.submitBlocks(...)

!! failure !!
pushPullBlocks(...) || pullBlocks(...)
*/
