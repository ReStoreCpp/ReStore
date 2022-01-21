
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <optional>
#include <random>
#include <vector>

#include "restore/core.hpp"
#include "restore/equal_load_balancer.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"
#include "restore/restore_vector.hpp"
#include "restore/timer.hpp"
#include "restore/two_phase_commit.hpp"

// TODO Pick initial centers not only from the data on the first rank
// TODO Use asynchronous MPI calls

namespace kmeans {


// kMeansData class to abstract away the handling of the multi-dimensional data
template <class data_t>
class kMeansData {
    public:
    // A Special object to pass to kMeansData::operator<< to signalize that all dimensions of the current data point
    // have been pushed. This enables us to check if all dimensions of each data point has been pushed.
    struct FinalizeDataPoint {
        bool tag;
    };

    // Constructor, takes a rvalue reference to a data vector and the number of dimensions
    template <typename T>
    kMeansData(T&& data, uint64_t numDimensions) : _data(std::forward<T>(data)),
                                                   _numDimensions(numDimensions) {
        if (_numDimensions == 0) {
            throw std::invalid_argument("I don't know how to handle 0 dimensions, sorry.");
        } else if (_data.size() % _numDimensions != 0) {
            throw std::invalid_argument(
                "The size of the data vector is not evenly divisible by the number of dimensions.");
        }
    }

    kMeansData(std::initializer_list<data_t> initializerList, uint64_t numDimensions)
        : _data(initializerList),
          _numDimensions(numDimensions) {
        if (_numDimensions == 0) {
            throw std::invalid_argument("I don't know how to handle 0 dimensions, sorry.");
        } else if (_data.size() % _numDimensions != 0) {
            throw std::invalid_argument(
                "The size of the data vector is not evenly divisible by the number of dimensions.");
        }
    }

    // Constructs a kMeansData object with the given number of dimensions and no data points.
    explicit kMeansData(uint64_t numDimensions) : _data(0), _numDimensions(numDimensions) {
        if (numDimensions == 0) {
            throw std::invalid_argument("I don't know how to handle a zero-dimensional space, sorry.");
        }
    }

    // Constructs a kMeansData object with the given number of dimensions and numDataPoints data points with all
    // dimensions set to initialValue.
    kMeansData(uint64_t numDataPoints, uint64_t numDimensions, data_t initialValue)
        : _data(numDataPoints * numDimensions, initialValue),
          _numDimensions(numDimensions) {
        if (numDimensions == 0) {
            throw std::invalid_argument("I don't know how to handle a zero-dimensional space, sorry.");
        }
    }

    // Returns the number of dimensions each data point in this object has.
    uint64_t numDimensions() const {
        assert(!valid() || _data.size() % _numDimensions == 0);
        return _numDimensions;
    }

    // Returns the number of data points in this object.
    uint64_t numDataPoints() const {
        assert(!valid() || _data.size() % _numDimensions == 0);
        return _data.size() / _numDimensions;
    }

    // Push back one dimension of a data point. After each dimension of this data point are pushed, finalize the data
    // point by calling this function with the special struct FinalizeDataPoint.
    kMeansData& operator<<(data_t element) {
        if (_dimensionsPushed >= _numDimensions) {
            throw std::invalid_argument(
                "Trying to push more dimensions than this data type has. Please push kMeansData::FinalizeDataPoint");
        }

        _dimensionsPushed++;
        _partialDataPoints = true;
        _data.push_back(element);

        return *this;
    }

    // Finalizes the current data point and checks that each of its dimension has been pushed.
    kMeansData& operator<<(FinalizeDataPoint _) {
        UNUSED(_);
        if (_dimensionsPushed != _numDimensions) {
            throw std::invalid_argument(
                "Trying to finalize a data point without pushing the proper number of dimensions first.");
        }
        _dimensionsPushed  = 0;
        _partialDataPoints = false;

        return *this;
    }

    // Resize this object to contain the given number of data points. The added data points will have all their
    // dimensions set to initialValue.
    void resize(size_t numDataPoints, data_t initialValue = 0) {
        _data.resize(numDataPoints * _numDimensions, initialValue);
    }

    void reserve(size_t numDataPoints) {
        _data.reserve(numDataPoints * _numDimensions);
    }

    // TODO We need a prettier interface for this
    const data_t getElementDimension(uint64_t dataIdx, uint64_t dimension) const {
        assert(dataIdx < numDataPoints());
        assert(dimension < numDimensions());
        return _data[dataIdx * numDimensions() + dimension];
    }

    data_t& getElementDimension(uint64_t dataIdx, uint64_t dimension) {
        assert(dataIdx < numDataPoints());
        assert(dimension < numDimensions());
        return _data[dataIdx * numDimensions() + dimension];
    }

    // Returns false if in an invalid state. This can for example happen if only some dimensions of a new data points
    // have been pushed or if not all data points have been added yet. Returns true otherwise.
    bool valid() const {
        // We can't use the numDataPoints() and numDimensions() getters, as these functions use valid() in their
        // assertions and this would therefore create an infinite loop.
        return !_partialDataPoints;
    }

    // Returns a pointer to the raw data.
    data_t* data() {
        return _data.data();
    }

    const std::vector<data_t>& dataVector() const {
        return _data;
    }

    std::vector<data_t>& dataVector() {
        return _data;
    }

    const data_t* data() const {
        return _data.data();
    }

    using const_iterator = typename std::vector<data_t>::const_iterator;
    using value_type     = data_t;
    const_iterator begin() const {
        return _data.cbegin();
    }

    const_iterator end() const {
        return _data.cend();
    }

    // Returns the number of elements in the underlying rad data vector.
    size_t dataSize() const {
        assert(numDimensions() * numDataPoints() == _data.size());
        return _data.size();
    }

    // Fills all dimensions of all elements with the specified value.
    void fill(data_t value) {
        std::fill(_data.begin(), _data.end(), value);
    }

    private:
    std::vector<data_t> _data;
    uint64_t            _numDimensions;
    uint64_t            _dimensionsPushed  = 0;
    bool                _partialDataPoints = false;
};

// k-means algorithm
template <class data_t, class MPI_Context>
class kMeansAlgorithm {
    static_assert(std::is_floating_point_v<data_t>, "Only floating point data types are supported, sorry.");

    public:
    // This struct represents an assignment of data points to centers. It includes information about which data point is
    // assigned to which center and how many data points are assigned to each center.
    struct PointToCenterAssignment {
        PointToCenterAssignment(uint64_t numDataPoints, uint64_t numCenters)
            : assignedCenter(numDataPoints),
              numPointsAssignedToCenter(numCenters, 0) {
            if (numDataPoints == 0 || numCenters == 0) {
                throw std::invalid_argument("Neither the number of data points nor the number of centers might be 0.");
            }
            assert(assignedCenter.size() == numDataPoints);
            assert(numPointsAssignedToCenter.size() == numCenters);
        }

        void resize(size_t numDataPoints) {
            assignedCenter.resize(numDataPoints, 0);
        }

        size_t numDataPoints() const {
            return assignedCenter.size();
        }

        uint64_t numCenters() const {
            return numPointsAssignedToCenter.size();
        }

        std::vector<size_t>   assignedCenter;
        std::vector<uint64_t> numPointsAssignedToCenter;
    };

    // TODO: Simplify these constructors using templating and forwarding
    // Constructor, takes a rvalue reference to data, which we take ownership of, the number of centers/clusters to
    // compute and the number of iterations to perform.
    kMeansAlgorithm(
        kMeansData<data_t>&& data, MPI_Context& mpiContext, bool faultTolerant, uint16_t replicationLevel,
        uint64_t blocksPerPermutationRange)
        : _data(std::move(data)),
          _centers(std::nullopt),
          _pointToCenterAssignment(std::nullopt),
          _mpiContext(mpiContext),
          _faultTolerant(faultTolerant),
          _reStoreWrapper(
              faultTolerant ? std::make_optional<ReStore::ReStoreVector<data_t>>(
                  numDimensions(), _mpiContext.getComm(), replicationLevel, blocksPerPermutationRange)
                            : std::nullopt),
          _loadBalancer(_initialBlockRanges(), _mpiContext.getCurrentSize()) {
        if (_data.numDataPoints() == 0) {
            throw std::invalid_argument("The data vector is empty -> No datapoints given.");
        } else if (!_data.valid()) {
            throw std::invalid_argument("The data object has to be in a valid state.");
        }

        // Submit the data points to the ReStore so we are able to recover them after a failure.
        if (faultTolerant) {
            try {
                TIME_PUSH_AND_START("submit-data"); // I tried to use TIME_BLOCK() here, but the compiler reordered the
                                                    // instructions and I timed nothing.
                const auto numBlocksLocal = _reStoreWrapper->submitData(_data.dataVector());
                assert(numBlocksLocal == _data.numDataPoints());
                UNUSED(numBlocksLocal);
                TIME_POP();
            } catch (std::runtime_error& e) {
                TIME_POP();
                throw e;
            }
        }
    }

    kMeansAlgorithm(
        std::initializer_list<data_t> initializerList, uint64_t numDimensions, MPI_Context& mpiContext,
        bool faultTolerant, uint16_t replicationLevel, uint64_t blocksPerPermutationRange)
        : kMeansAlgorithm(
            kMeansData<data_t>(initializerList, numDimensions), mpiContext, faultTolerant, replicationLevel,
            blocksPerPermutationRange) {}

    kMeansAlgorithm(
        std::vector<data_t>&& data, uint64_t numDimensions, MPI_Context& mpiContext, bool faultTolerant,
        uint16_t replicationLevel, uint64_t blocksPerPermutationRange)
        : kMeansAlgorithm(
            kMeansData<data_t>(std::move(data), numDimensions), mpiContext, faultTolerant, replicationLevel,
            blocksPerPermutationRange) {}

    // Sets the centers to the provided data, takes ownership of the data object. Must be called with the same
    // parameters on all ranks. Local operation, does therefore not report rank failures.
    template <typename T>
    void setCenters(T&& centers) {
        if constexpr (std::is_same_v<std::remove_reference_t<T>, kMeansData<data_t>>) {
            _centers.emplace(std::forward<T>(centers));
        } else {
            _centers.emplace(std::forward<T>(centers), numDimensions());
        }
    }

    void setCenters(std::initializer_list<data_t> initializerList) {
        if (initializerList.size() % numDimensions() != 0) {
            throw std::invalid_argument("Length of initializer list not evenly dividible by the number of dimensions.");
        }
        _centers.emplace(initializerList, numDimensions());
    }

    // Picks the initial centers. Will override the current centers. This is a global operation, and might therefore
    // report rank failures by throwing a FaultException.
    void pickCentersRandomly(uint64_t numCenters, unsigned long seed = 0) {
        if (numCenters == 0) {
            throw std::invalid_argument("I don't know how to handle 0 centers, sorry.");
        } else if (numCenters > numDataPoints()) {
            throw std::invalid_argument("Using more centers than data points is now allowed.");
        }

        // Reset the current centers
        _centers.emplace(numDimensions());

        assert(_centers->numDataPoints() == 0);
        if (_mpiContext.getMyCurrentRank() == 0) {
            std::mt19937                          generator(seed);
            std::uniform_int_distribution<size_t> randomDataPointIndex(0, numDataPoints() - 1);

            // Push back randomly picked data point into the centers data structure
            for (uint64_t center = 0; center < numCenters; center++) {
                auto dataPointIdx = randomDataPointIndex(generator);
                for (size_t dimension = 0; dimension < numDimensions(); dimension++) {
                    *_centers << _data.getElementDimension(dataPointIdx, dimension);
                }
                *_centers << typename kMeansData<data_t>::FinalizeDataPoint();
            }
        } else {
            _centers->resize(numCenters);
        }
        assert(_centers->numDataPoints() == numCenters);
        _mpiContext.broadcast(_centers->data(), _centers->dataSize()); // May throw a FaultException
    }

    // Returns the number of dimensions our data has. This is a local operation and will not report rank failures.
    uint64_t numDimensions() const {
        assert(!_centers.hasValue() || _centers->numDimensions() == _data.numDimensions());
        return _data.numDimensions();
    }

    // Returns the number of centers/clusters we use. This is a local operation and will not report rank failures.
    uint64_t numCenters() const {
        if (!_centers.hasValue()) {
            return 0;
        } else {
            assert(_centers->numDimensions() == _data.numDimensions());
            assert(_centers->numDataPoints() >= 0);
            return _centers->numDataPoints();
        }
    }

    // Return the number of data points we have. This is a local operations and will therefore not report rank failures.
    uint64_t numDataPoints() const {
        return _data.numDataPoints();
    }

    // Returns our current centers. This is a local operation and will not report rank failures.
    const kMeansData<data_t>& centers() const {
        return *_centers;
    }

    // Returns a reference to our data. Only valid for as long as this object lives. This is a local operation and will
    // therefore not report rank failures.
    const kMeansData<data_t>& data() const {
        return _data;
    }

    // Return the points to center assignment. This is a local operation and will therefore not report rank failures.
    const PointToCenterAssignment& pointToCenterAssignment() const {
        assert(_pointToCenterAssignment);
        return *_pointToCenterAssignment;
    }

    // Assign each (local) data point to the closest center. This is a local operation and will therefore not report
    // rank failures. Consider using performIterations(...) if you have no special needs.
    // We assume that all data points with an index < startIndex are allready assigned to their respective centers. Use
    // a startIndex > 0 for example after restoring data which resided on a failed rank and is thus new to this rank.
    void assignPointsToCenters(uint64_t startIndex = 0) {
        assert(_data.valid());

        if (numDataPoints() == 0) {
            throw std::runtime_error("I don't have any data points.");
        } else if (numCenters() == 0) {
            throw std::runtime_error("I don't have any cluster centers.");
        } else if (numDimensions() == 0) {
            throw std::runtime_error("I have zero-dimensional data.");
        } else if (startIndex >= numDataPoints()) {
            throw std::out_of_range("You requested to start at a non-existing data point.");
        }

        // Engage data structure if it is not already.
        assert(_pointToCenterAssignment || startIndex == 0);
        if (!_pointToCenterAssignment) {
            _pointToCenterAssignment.emplace(numDataPoints(), numCenters());
        } else if (startIndex == 0) { // else, reset the number of points assigned to the centers
            assert(_pointToCenterAssignment);
            assert(_pointToCenterAssignment->numDataPoints() >= numDataPoints());
            std::fill(
                _pointToCenterAssignment->numPointsAssignedToCenter.begin(),
                _pointToCenterAssignment->numPointsAssignedToCenter.end(), 0);
            // Resetting the assignments is not necessary as these are overwritten anyway
        }
        assert(_pointToCenterAssignment);

        // If we got new data since the last time we were called, we need to increase the size of the assignedCenter
        // data structure.
        if (_pointToCenterAssignment->numDataPoints() < numDataPoints()) {
            _pointToCenterAssignment->resize(numDataPoints());
        }

        // Compute the distance between each data point and each center, assign each point to it's closest cluster
        // center.
        assert(_pointToCenterAssignment->numDataPoints() == numDataPoints());
        assert(_pointToCenterAssignment->numCenters() == numCenters());
        for (uint64_t dataIdx = startIndex; dataIdx < numDataPoints(); dataIdx++) {
            size_t closestCenter           = std::numeric_limits<size_t>::max();
            data_t distanceToClosestCenter = std::numeric_limits<data_t>::max();
            for (size_t centerIdx = 0; centerIdx < numCenters(); centerIdx++) {
                data_t distanceToCenter = 0;
                for (size_t dimension = 0; dimension < numDimensions(); dimension++) {
                    auto delta = _data.getElementDimension(dataIdx, dimension)
                                 - _centers->getElementDimension(centerIdx, dimension);
                    distanceToCenter += delta * delta;
                }
                // We don't need to comput the square root, as we are only comparing the distance and sqrt is strictly
                // monotonic.
                // distanceToCenter = std::sqrt(distanceToCenter);
                if (distanceToCenter < distanceToClosestCenter) {
                    closestCenter           = centerIdx;
                    distanceToClosestCenter = distanceToCenter;
                }
                assert(distanceToClosestCenter >= 0);
            }
            assert(closestCenter != std::numeric_limits<size_t>::max());

            _pointToCenterAssignment->assignedCenter[dataIdx] = closestCenter;
        }
        assert(_pointToCenterAssignment->assignedCenter.size() == numDataPoints());

        // Compute how many points are assigned to each center.
        assert(std::for_each(
            _pointToCenterAssignment->numPointsAssignedToCenter.begin(),
            _pointToCenterAssignment->numPointsAssignedToCenter.end(),
            [](uint64_t numPoints) { return numPoints == 0; }));

        assert(_pointToCenterAssignment->numPointsAssignedToCenter.size() == numCenters());
        for (uint64_t dataIdx = startIndex; dataIdx < numDataPoints(); dataIdx++) {
            assert(dataIdx < _pointToCenterAssignment->assignedCenter.size());
            auto assignedCenter = _pointToCenterAssignment->assignedCenter[dataIdx];

            assert(assignedCenter < _pointToCenterAssignment->numPointsAssignedToCenter.size());
            _pointToCenterAssignment->numPointsAssignedToCenter[assignedCenter]++;
        }
    }

    // Do one update round of the center positions with the current data. This is a global operation and might throw
    // FaultExceptions. Consider using performIterations(...) if you have no special needs.
    void updateCenters(bool onlyNewData = false) {
        assert(_data.valid());
        if (numDataPoints() == 0) {
            throw std::runtime_error("I don't have any data points.");
        } else if (numCenters() == 0) {
            throw std::runtime_error("I don't have any cluster centers.");
        }

        uint64_t startIndex;
        if (!_contribToCenterPosition) {
            if (onlyNewData) {
                throw std::runtime_error("onlyNewData = true is not allowed when first calling updateCenters.");
            }
            _contribToCenterPosition.emplace(numCenters(), numDimensions(), 0);
            startIndex = 0;
        } else {
            if (!onlyNewData) {
                _contribToCenterPosition->fill(0);
                startIndex = 0;
            } else {
                // Do not reset the contributions.
                if (numDataPoints() - 1 == _contribLastIndex) {
                    // No new data for us but we still have to participate in the collectives and contribute our data.
                    startIndex = numDataPoints();
                } else {
                    startIndex = _contribLastIndex + 1;
                    assert(startIndex < numDataPoints());
                }
            }
        }

        // Compute contribution of local data points to the positions of the new centers.
        for (size_t dataIdx = startIndex; dataIdx < numDataPoints(); dataIdx++) {
            auto centerIdx = _pointToCenterAssignment->assignedCenter[dataIdx];
            for (uint64_t dimension = 0; dimension < numDimensions(); dimension++) {
                _contribToCenterPosition->getElementDimension(centerIdx, dimension) +=
                    _data.getElementDimension(dataIdx, dimension);
            }
        }
        _contribLastIndex = numDataPoints() - 1;

        // Allreduce the local contributions to the center positions
        _mpiContext.allreduce(_contribToCenterPosition->data(), MPI_SUM, _contribToCenterPosition->dataSize());
        // TODO Think about what would happen if we detect a rank failure during this call.
        _mpiContext.allreduce(_pointToCenterAssignment->numPointsAssignedToCenter, MPI_SUM);
        assert(_pointToCenterAssignment->numPointsAssignedToCenter.size() == numCenters());
        assert(_contribToCenterPosition->numDataPoints() == numCenters());

        // Calculate new center positions
        for (size_t centerIdx = 0; centerIdx < numCenters(); centerIdx++) {
            auto numPointsAssignedToThisCenter =
                static_cast<data_t>(_pointToCenterAssignment->numPointsAssignedToCenter[centerIdx]);
            if (numPointsAssignedToThisCenter > 0) {
                for (size_t dimension = 0; dimension < numDimensions(); dimension++) {
                    _centers->getElementDimension(centerIdx, dimension) =
                        _contribToCenterPosition->getElementDimension(centerIdx, dimension)
                        / numPointsAssignedToThisCenter;
                }
            } // If no point is assigned to this center, we leave the center where it is.
        }
    }

    // Perform the number of k-means iterations provided. Rank failures will be automatically handled.
    // Passing numIterations = 0 does nothing.
    void performIterations(const uint64_t numIterations) {
        assert(_data.valid());

        // Perform numIterations k-means iterations
        uint64_t iteration = 0;
        while (iteration < numIterations) {
            try {
                // Assign the points to centers. Calling this again after a failure won't do any harm.
                assignPointsToCenters();

                // Compute contribution of local data points to the positions of the new centers
                updateCenters();

                if (_faultTolerant) {
                    TIME_PUSH_AND_START("checkpoint-creation");
                    // Has everyone completed this iteration without detecting a rank failure?
                    _mpiContext.ft_barrier();

                    // Nobody failed, commit to this iteration's changes.
                    _centers.commit();
                    TIME_POP();
                }
                iteration++;
            } catch (typename ReStoreMPI::FaultException& e) {
                if (!_faultTolerant) {
                    throw e;
                }
                // TODO: Handle failures during recovery. For now, we decided to not simulate that.

                // Roll back to the latest checkpoint of the center positions, this is a local operation and therefore
                // works even with a broken communicator.
                // Multiple calls to .rollback() will all roll back to the same (most recent) checkpoint.
                TIME_PUSH_AND_START("center-rollback");
                _centers.rollback();

                // Fix the communicator
                TIME_NEXT_SECTION("fix-communicator");
                _mpiContext.fixComm();
                _reStoreWrapper->updateComm(_mpiContext.getComm());

                // Until we commit to this new data distribution, calling the follwing function twice will return the
                // same data.
                TIME_NEXT_SECTION("rebalance-after-failure");
                auto newBlocks = _loadBalancer.getNewBlocksAfterFailureForPullBlocks(
                    _mpiContext.getRanksDiedSinceLastCall(), _mpiContext.getMyOriginalRank());

                // ReStore the input data that resided on the failed ranks.
                TIME_NEXT_SECTION("restore-data");
                auto numElementsBeforeRestore = _data.numDataPoints();
                _reStoreWrapper->restoreDataAppendPullBlocks(_data.dataVector(), newBlocks);

                TIME_NEXT_SECTION("reassign-points-after-failure");
                // If we got new data points, assign them to the cluster centers.
                // This function will also update the local data structures it needs to reflect the new data
                // distribution
                if (numDataPoints() > numElementsBeforeRestore) {
                    assignPointsToCenters(numElementsBeforeRestore);
                }

                // Collectively update the center positions.
                TIME_NEXT_SECTION("update-centers-after-failure");
                updateCenters(true);

                // Has everyone completed the restoration successfully?
                TIME_NEXT_SECTION("commit-to-restoration");
                _mpiContext.ft_barrier();

                // Everyone completed the restoration successfully, we can commit to the new data distribution.
                _loadBalancer.commitToPreviousCall();
                iteration++;

                TIME_POP();
            }
        }
    }

    // TODO Handle rank failures, in this function and in the assignment points to centers. (Currently, the points have
    // no ids and are redistributed upon a failure. This means we have no way of identifying them after a failure.)
    std::vector<uint64_t> collectClusterAssignments() {
        if (!_pointToCenterAssignment) {
            throw std::runtime_error(
                "I don't have any point-to-center assignments. Please perform an interation first.");
        }

        return _mpiContext.gatherv(_pointToCenterAssignment->assignedCenter);
    }

    private:
    using BlockRange     = std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::original_rank_t>;
    using BlockRangeList = std::vector<BlockRange>;

    BlockRangeList _initialBlockRanges() {
        auto myNumBlocks = _data.numDataPoints();
        auto myRank      = _mpiContext.getMyCurrentRank();

        auto myFirstBlockId = _mpiContext.exclusive_scan(myNumBlocks, MPI_SUM);

        BlockRange myBlockRange = std::make_pair(std::make_pair(myFirstBlockId, myNumBlocks), myRank);
        return _mpiContext.allgather(myBlockRange);
    }

    kMeansData<data_t> _data;
    // Will be empty after construction, before the initial centers are picked.
    TwoPhaseCommit<kMeansData<data_t>>     _centers;
    std::optional<PointToCenterAssignment> _pointToCenterAssignment;
    // This array accumulates the contributions of the data points assigned to a center to that center.
    std::optional<kMeansData<data_t>>             _contribToCenterPosition;
    uint64_t                                      _contribLastIndex = 0;
    MPI_Context&                                  _mpiContext;
    bool                                          _faultTolerant;
    std::optional<ReStore::ReStoreVector<data_t>> _reStoreWrapper;
    ReStore::EqualLoadBalancer                    _loadBalancer;
}; // namespace kmeans

template <class data_t>
kMeansData<data_t> generateRandomData(size_t numDataPoints, uint64_t numDimensions, unsigned long seed = 0) {
    static_assert(std::is_floating_point_v<data_t>, "Only floating point data types are supported.");

    kMeansData<data_t> data(numDimensions);

    std::mt19937 generator(seed);
    // std::uniform_real_distribution<data_t> dataDistribution(
    //    std::numeric_limits<data_t>::min(), std::numeric_limits<data_t>::max());
    std::uniform_real_distribution<data_t> dataDistribution(-10000, 10000);

    for (size_t dataPoint = 0; dataPoint < numDataPoints; dataPoint++) {
        for (size_t dimension = 0; dimension < numDimensions; dimension++) {
            data << (dataDistribution(generator));
        }
        data << typename kMeansData<data_t>::FinalizeDataPoint();
    }
    assert(data.valid());

    return data;
}

template <class data_t>
kMeansData<data_t> loadDataFromFile(const std::string& inputFile) {
    std::ifstream file(inputFile);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + inputFile + " for reading.");
    }

    uint64_t numDataPoints = std::numeric_limits<uint64_t>::max();
    uint16_t numDimensions = std::numeric_limits<uint16_t>::max();
    file >> numDataPoints;
    file >> numDimensions;

    assert(numDataPoints < std::numeric_limits<uint64_t>::max());
    assert(numDimensions < std::numeric_limits<uint16_t>::max());
    if (numDataPoints == 0 || numDimensions == 0) {
        throw std::runtime_error("Number of data points and dimensions may not be zero zero.");
    }

    kMeansData<data_t> data(numDimensions);
    data.reserve(numDataPoints);
    for (uint64_t dataPoint = 0; dataPoint < numDataPoints; dataPoint++) {
        for (uint16_t dimension = 0; dimension < numDimensions; dimension++) {
            data_t value;
            file >> value;
            data << value;
        }
        data << typename kMeansData<data_t>::FinalizeDataPoint();
    }

    return data;
}

template <class data_t>
void writeDataToFile(kMeansData<data_t> data, const std::string& outputFile) {
    // Open the file for writing.
    std::ofstream file(outputFile);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + outputFile + " for writing.");
    }

    // Write the number of data points and the number of dimensions.
    file << data.numDataPoints() << " " << data.numDimensions() << std::endl;

    // Write the data points.
    for (size_t dataPoint = 0; dataPoint < data.numDataPoints(); dataPoint++) {
        for (size_t dimension = 0; dimension < data.numDimensions(); dimension++) {
            file << data.getElementDimension(dataPoint, dimension);
            if (dimension < data.numDimensions() - 1) {
                file << " ";
            } else if (dataPoint < data.numDataPoints() - 1) {
                file << std::endl;
            }
        }
    }
}
} // namespace kmeans
