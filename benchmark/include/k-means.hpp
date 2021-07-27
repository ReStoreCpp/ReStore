
#include <array>
#include <cstdint>
#include <mpi.h>
#include <optional>
#include <random>
#include <vector>

#include "restore/helpers.hpp"

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

        std::vector<size_t>   assignedCenter;
        std::vector<uint64_t> numPointsAssignedToCenter;
    };

    // TODO: Simplify these constructors using templating and forwarding
    // Constructor, takes a rvalue reference to data, which we take ownership of, the number of centers/clusters to
    // compute and the number of iterations to perform.
    kMeansAlgorithm(kMeansData<data_t>&& data, MPI_Context& mpiContext)
        : _data(std::move(data)),
          _centers(std::nullopt),
          _pointToCenterAssignment(std::nullopt),
          _mpiContext(mpiContext) {
        if (_data.numDataPoints() == 0) {
            throw std::invalid_argument("The data vector is empty -> No datapoints given.");
        } else if (!_data.valid()) {
            throw std::invalid_argument("The data object has to be in a valid state.");
        }
    }

    // Constructor
    kMeansAlgorithm(
        std::initializer_list<data_t> initializerList, uint64_t numDimensions, MPI_Context& mpiContext)
        : _data(initializerList, numDimensions),
          _centers(std::nullopt),
          _pointToCenterAssignment(std::nullopt),
          _mpiContext(mpiContext) {
        if (_data.numDataPoints() == 0) {
            throw std::invalid_argument("The data vector is empty -> No datapoints given.");
        } else if (!_data.valid()) {
            throw std::invalid_argument("The data object has to be in a valid state.");
        }
    }

    // Constructor, takes a rvalue reference to data, which we take ownership of, the number of centers/clusters to
    // compute and the number of iterations to perform.
    kMeansAlgorithm(std::vector<data_t>&& data, uint64_t numDimensions, MPI_Context& mpiContext)
        : _data(std::move(data), numDimensions),
          _centers(std::nullopt),
          _pointToCenterAssignment(std::nullopt),
          _mpiContext(mpiContext) {
        if (_data.numDataPoints() == 0) {
            throw std::invalid_argument("The data vector is empty -> No datapoints given.");
        } else if (!_data.valid()) {
            throw std::invalid_argument("The data object has to be in a valid state.");
        }
    }

    // Sets the centers to the provided data, takes ownership of the data object. Must be called with the same
    // parameters on all ranks.
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

    // Picks the initial centers. Will override the current centers
    void pickCentersRandomly(uint64_t numCenters) {
        if (numCenters == 0) {
            throw std::invalid_argument("I don't know how to handle 0 centers, sorry.");
        } else if (numCenters > numDataPoints()) {
            throw std::invalid_argument("Using more centers than data points is now allowed.");
        }

        // Reset the current centers
        _centers.emplace(numDimensions());

        assert(_centers->numDataPoints() == 0);
        if (_mpiContext.getMyCurrentRank() == 0) {
            std::random_device                    randomDevice;
            std::mt19937                          generator(randomDevice());
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
        _mpiContext.broadcast(_centers->data(), _centers->dataSize());
    }

    // Returns the number of dimensions our data has.
    uint64_t numDimensions() const {
        assert(!_centers || _centers->numDimensions() == _data.numDimensions());
        return _data.numDimensions();
    }

    // Returns the number of centers/clusters we use.
    uint64_t numCenters() const {
        if (!_centers) {
            return 0;
        } else {
            assert(_centers->numDimensions() == _data.numDimensions());
            assert(_centers->numDataPoints() >= 0);
            return _centers->numDataPoints();
        }
    }

    // Return the number of data points we have.
    uint64_t numDataPoints() const {
        return _data.numDataPoints();
    }

    // Returns our current centers
    const kMeansData<data_t>& centers() const {
        return *_centers;
    }

    // Returns a reference to our data. Only valid for as long as this object lives.
    const kMeansData<data_t>& data() const {
        return _data;
    }

    // Return the points to center assignment
    const PointToCenterAssignment& pointToCenterAssignment() const {
        return *_pointToCenterAssignment;
    }

    // Assign each data point to the closest center
    // Consider using performIterations(...) if you have no special needs.
    void assignPointsToCenters() {
        assert(_data.valid());

        if (numDataPoints() == 0) {
            throw std::runtime_error("I don't have any data points.");
        } else if (numCenters() == 0) {
            throw std::runtime_error("I don't have any cluster centers.");
        }

        // Engage data structure if it is not already
        if (!_pointToCenterAssignment) {
            _pointToCenterAssignment.emplace(numDataPoints(), numCenters());
        } else { // else, reset the number of points assigned to the centers
            assert(_pointToCenterAssignment);
            std::fill(
                _pointToCenterAssignment->numPointsAssignedToCenter.begin(),
                _pointToCenterAssignment->numPointsAssignedToCenter.end(), 0);
            // Resetting the assignments is not necessary as these are overwritten anyway
        }

        for (uint64_t dataIdx = 0; dataIdx < numDataPoints(); dataIdx++) {
            size_t closestCenter           = std::numeric_limits<size_t>::max();
            data_t distanceToClosestCenter = std::numeric_limits<data_t>::max();
            for (size_t centerIdx = 0; centerIdx < numCenters(); centerIdx++) {
                data_t distanceToCenter = 0;
                for (size_t dimension = 0; dimension < numDimensions(); dimension++) {
                    auto delta = _data.getElementDimension(dataIdx, dimension)
                                 - _centers->getElementDimension(centerIdx, dimension);
                    distanceToCenter += delta * delta;
                }
                // TODO We probably don't need this
                distanceToCenter = std::sqrt(distanceToCenter);

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

        for (auto assignedCenter: _pointToCenterAssignment->assignedCenter) {
            _pointToCenterAssignment->numPointsAssignedToCenter[assignedCenter]++;
        }
        assert(_pointToCenterAssignment->numPointsAssignedToCenter.size() == numCenters());
    }

    // Do one update round of the center positions with the current data.
    // Consider using performIterations(...) if you have no special needs.
    void updateCenters() {
        assert(_data.valid());
        if (numDataPoints() == 0) {
            throw std::runtime_error("I don't have any data points.");
        } else if (numCenters() == 0) {
            throw std::runtime_error("I don't have any cluster centers.");
        }

        // This array accumulates the contributions of the data points assigned to a center to that center.
        kMeansData<data_t> contribToCenterPosition(numCenters(), numDimensions(), 0);

        // Compute contribution of local data points to the positions of the new centers
        for (size_t dataIdx = 0; dataIdx < numDataPoints(); dataIdx++) {
            auto centerIdx = _pointToCenterAssignment->assignedCenter[dataIdx];
            for (uint64_t dimension = 0; dimension < numDimensions(); dimension++) {
                contribToCenterPosition.getElementDimension(centerIdx, dimension) +=
                    _data.getElementDimension(dataIdx, dimension);
            }
        }

        // Allreduce the local contributions to the center positions
        _mpiContext.allreduce(contribToCenterPosition.data(), MPI_SUM, contribToCenterPosition.dataSize());
        _mpiContext.allreduce(_pointToCenterAssignment->numPointsAssignedToCenter, MPI_SUM);
        assert(_pointToCenterAssignment->numPointsAssignedToCenter.size() == numCenters());
        assert(contribToCenterPosition.numDataPoints() == numCenters());

        // Calculate new center positions
        for (size_t centerIdx = 0; centerIdx < numCenters(); centerIdx++) {
            auto numPointsAssignedToThisCenter =
                static_cast<data_t>(_pointToCenterAssignment->numPointsAssignedToCenter[centerIdx]);
            if (numPointsAssignedToThisCenter > 0) {
                for (size_t dimension = 0; dimension < numDimensions(); dimension++) {
                    _centers->getElementDimension(centerIdx, dimension) =
                        contribToCenterPosition.getElementDimension(centerIdx, dimension)
                        / numPointsAssignedToThisCenter;
                }
            } // If no point is assigned to this center, we leave the center where it is.
        }
    }

    // Perform the number of k-means iterations provided.
    // Passing numIterations = 0 does nothing
    void performIterations(uint64_t numIterations) {
        assert(_data.valid());

        // Perform numIterations k-means iterations
        for (uint64_t iteration = 0; iteration < numIterations; iteration++) {
            // Assign the points to centers
            assignPointsToCenters();

            // Compute contribution of local data points to the positions of the new centers
            updateCenters();
        }
    }

    std::vector<uint64_t> collectClusterAssignments() {
        if (!_pointToCenterAssignment) {
            throw std::runtime_error(
                "I don't have any point-to-center assignments. Please perform an interation first.");
        }

        return _mpiContext.gatherv(_pointToCenterAssignment->assignedCenter);
    }

    private:
    kMeansData<data_t> _data;
    // Will be empty after construction, before the initial centers are picked.
    std::optional<kMeansData<data_t>>      _centers;
    std::optional<PointToCenterAssignment> _pointToCenterAssignment;
    MPI_Context&                     _mpiContext;
};

template <class data_t>
kMeansData<data_t> generateRandomData(size_t numDataPoints, uint64_t numDimensions) {
    static_assert(std::is_floating_point_v<data_t>, "Only floating point data types are supported.");

    kMeansData<data_t> data(numDimensions);

    std::random_device                     randomDevice;
    std::mt19937                           generator(randomDevice());
    std::uniform_real_distribution<data_t> dataDistribution(
        std::numeric_limits<data_t>::min(), std::numeric_limits<data_t>::max());

    for (size_t dataPoint = 0; dataPoint < numDataPoints; dataPoint++) {
        for (size_t dimension = 0; dimension < numDimensions; dimension++) {
            data << (dataDistribution(generator));
        }
        data << typename kMeansData<data_t>::FinalizeDataPoint();
    }
    assert(data.valid());

    return data;
}
} // namespace kmeans