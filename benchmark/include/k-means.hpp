
#include <array>
#include <cstdint>
#include <mpi.h>
#include <random>
#include <vector>

#include "restore/helpers.hpp"

// TODO Pick initial centers not only from the data on the first rank
// TODO Abstract away MPI functionality
// TODO Accept number of data points that are not evenly divisible by the number of ranks
// TODO Use asynchronous MPI calls
// TODO Code Cleanup!
// TODO Add assertions
// TODO User proper iterators

namespace kmeans {

// ----- MPI Helpers -----

// Returns the rank of this MPI process in MPI_COMM_WORLD
size_t rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return asserting_cast<size_t>(rank);
}

// Returns the number of MPI ranks in MPI_COMM_WORLD
size_t numRanks() {
    int numRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    return asserting_cast<size_t>(numRanks);
}

// kMeansData class to abstract away the handling of the multi-dimensional data
template <class data_t>
class kMeansData {
    public:
    // A Special object to pass to kMeansData::operator<< to signalize that all dimensions of the current data point
    // have been pushed. This enables us to check if all dimensions of each data point has been pushed.
    struct FinalizeDataPoint {
        bool tag;
    };

    // TODO Do we need this constructor?
    // Constructor, takes a rvalue reference to a data vector and the number of dimensions
    // kMeansData(std::vector<data_t>&& data, uint64_t numDimensions)
    //     : _data(std::move(data)),
    //       _numDimensions(numDimensions) {
    //     if (_data.size() % _numDimensions != 0) {
    //         throw std::runtime_error(
    //             "The size of the data vector is not evenly divisible by the number of dimensions.");
    //     }
    //     _numDataPoints = _data.size() % _numDimensions;
    // }

    // Constructs a kMeansData object with the given number of dimensions and no data points.
    explicit kMeansData(uint64_t numDimensions) : _data(0), _numDimensions(numDimensions), _numDataPoints(0) {
        if (numDimensions == 0) {
            throw std::runtime_error("I don't know how to handle a zero-dimensional space, sorry.");
        }
    }

    // Constructs a kMeansData object with the given number of dimensions and numDataPoints data points with all
    // dimensions set to initialValue.
    kMeansData(uint64_t numDataPoints, uint64_t numDimensions, data_t initialValue)
        : _data(numDataPoints * numDimensions, initialValue),
          _numDimensions(numDimensions),
          _numDataPoints(numDataPoints) {
        if (numDimensions == 0) {
            throw std::runtime_error("I don't know how to handle a zero-dimensional space, sorry.");
        }
    }

    // Returns the number of dimensions each data point in this object has.
    uint64_t numDimensions() const {
        assert(!valid() || _numDataPoints == _data.size() / _numDimensions);
        return _numDimensions;
    }

    // Returns the number of data points in this object.
    uint64_t numDataPoints() const {
        assert(!valid() || _numDataPoints == _data.size() / _numDimensions);
        return _numDataPoints;
    }

    // Push back one dimension of a data point. After each dimension of this data point are pushed, finalize the data
    // point by calling this function with the special struct FinalizeDataPoint.
    kMeansData& operator<<(data_t element) {
        if (_dimensionsPushed >= _numDimensions) {
            throw std::runtime_error(
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
            throw std::runtime_error(
                "Trying to finalize a data point without pushing the proper number of dimensions first.");
        }
        _dimensionsPushed = 0;
        _numDataPoints++;
        _partialDataPoints = false;

        return *this;
    }

    // Resize this object to contain the given number of data points. The added data points will have all their
    // dimensions set to initialValue.
    void resize(size_t numDataPoints, data_t initialValue = 0) {
        _numDataPoints = numDataPoints;
        _data.resize(_numDataPoints * _numDimensions, initialValue);
    }

    // TODO We need a prettier interface for this
    // Access the underlying vector object, this is ugly and should be changed
    const data_t operator[](size_t idx) const {
        return _data[idx];
    }

    // Access the underlying vector object, this is ugly and should be changed
    data_t& operator[](size_t idx) {
        return _data[idx];
    }

    // Returns false if in an invalid state. This can for example happen if only some dimensions of a new data points
    // have been pushed or if not all data points have been added yet. Returns true otherwise.
    bool valid() const {
        // We can't use the numDataPoints() and numDimensions() getters, as these functions use valid() in their
        // assertions and this would therefore create an infinite loop.
        return !_partialDataPoints && _data.size() == _numDataPoints * _numDimensions;
    }

    // Returns a pointer to the raw data.
    data_t* data() {
        return _data.data();
    }

    // Returns the number of elements in the underlying rad data vector.
    size_t dataSize() const {
        assert(numDimensions() * numDataPoints() == _data.size());
        return _data.size();
    }

    private:
    std::vector<data_t> _data;
    uint64_t            _numDimensions;
    uint64_t            _dimensionsPushed = 0;
    uint64_t            _numDataPoints;
    bool                _partialDataPoints = false;
};

// k-means algorithm
// TODO add more comments
template <class data_t>
class kMeansAlgorithm {
    static_assert(std::is_floating_point_v<data_t>, "Only floating point data types are supported, sorry.");

    public:
    kMeansAlgorithm(kMeansData<data_t>&& data, uint64_t numCenters, uint64_t numIterations)
        : _data(std::move(data)),
          _centers(data.numDimensions()),
          _pointToCenterAssignment(_data.numDataPoints(), numCenters),
          _numIterations(numIterations),
          _numCenters(numCenters) {
        if (data.numDataPoints() == 0) {
            throw std::runtime_error("The data vector is empty -> No datapoints given.");
        } else if (!_data.valid()) {
            throw std::runtime_error("The data object has to be in a valid state.");
        }
    }

    void pickInitialCenters() {
        assert(_numCenters > 0);
        assert(_centers.numDataPoints() == 0);

        if (rank() == 0) {
            std::random_device                    randomDevice;
            std::mt19937                          generator(randomDevice());
            std::uniform_int_distribution<size_t> indexDistribution(0, numDataPoints());

            for (uint64_t center = 0; center < numCenters(); center++) {
                auto index = indexDistribution(generator);
                for (size_t dimension = 0; dimension < numDimensions(); dimension++) {
                    _centers << _data[index + dimension];
                }
                _centers << typename kMeansData<data_t>::FinalizeDataPoint();
            }
        } else {
            _centers.resize(numCenters());
        }
        assert(_centers.numDataPoints() == _numCenters);
    }

    uint64_t numDimensions() const {
        assert(_centers.numDimensions() == _data.numDimensions());
        return _data.numDimensions();
    }

    uint64_t numCenters() const {
        assert(_centers.numDimensions() == _data.numDimensions());
        assert(_centers.numDataPoints() >= 0 && _centers.numDataPoints() <= _numCenters);
        return _numCenters;
    }

    uint64_t numDataPoints() const {
        return _data.numDataPoints();
    }

    const kMeansData<data_t>& centers() const {
        return _centers;
    }

    const kMeansData<data_t>& data() const {
        return _data;
    }

    struct PointToCenterAssignment {
        PointToCenterAssignment(uint64_t numDataPoints, uint64_t numCenters)
            : assignedCenter(numDataPoints),
              numPointsAssignedToCenter(numCenters, 0) {
            if (numDataPoints == 0 || numCenters == 0) {
                throw std::runtime_error("Neither the number of data points nor the number of centers might be 0.");
            }
            assert(assignedCenter.size() == numDataPoints);
            assert(numPointsAssignedToCenter.size() == numCenters);
        }

        std::vector<size_t>   assignedCenter;
        std::vector<uint64_t> numPointsAssignedToCenter;
    };

    // Assign each data point to the closest center
    void assignPointsToCenters() {
        assert(_data.valid());

        for (uint64_t dataIdx = 0; dataIdx < numDataPoints(); dataIdx++) {
            size_t closestCenter           = 0;
            float  distanceToClosestCenter = std::numeric_limits<float>::max();
            for (size_t centerIdx = 0; centerIdx < numCenters(); centerIdx++) {
                float distanceToCenter = 0;
                for (size_t dimension = 0; dimension < numDimensions(); dimension++) {
                    float dataF   = _data[dataIdx];
                    float centerF = _centers[centerIdx];
                    distanceToCenter += (dataF - centerF) * (dataF - centerF);
                }
                distanceToCenter = std::sqrt(distanceToCenter);

                if (distanceToCenter < distanceToClosestCenter) {
                    closestCenter           = centerIdx;
                    distanceToClosestCenter = distanceToCenter;
                }
            }

            _pointToCenterAssignment.assignedCenter[dataIdx] = closestCenter;
        }
        assert(_pointToCenterAssignment.assignedCenter.size() == numDataPoints());

        for (auto assignedCenter: _pointToCenterAssignment.assignedCenter) {
            _pointToCenterAssignment.numPointsAssignedToCenter[assignedCenter]++;
        }
        assert(_pointToCenterAssignment.numPointsAssignedToCenter.size() == numCenters());
    }

    void computeNewCenters() {
        assert(_data.valid());

        kMeansData<data_t> contribToCenterPosition(numDataPoints(), numDimensions(), 0);

        // Compute contribution of local data points to the positions of the new centers
        for (size_t dataIdx = 0; dataIdx < numDataPoints(); dataIdx++) {
            auto centerIdx = _pointToCenterAssignment.assignedCenter[dataIdx];
            for (uint64_t dimension = 0; dimension < numDimensions(); dimension++) {
                contribToCenterPosition[dataIdx + dimension] += _data[dataIdx + dimension];
            }
            _pointToCenterAssignment.numPointsAssignedToCenter[centerIdx]++;
        }

        // Allreduce the local contributions to the center positions
        MPI_Allreduce(
            MPI_IN_PLACE, contribToCenterPosition.data(), asserting_cast<int>(contribToCenterPosition.dataSize()),
            MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(
            MPI_IN_PLACE, _pointToCenterAssignment.numPointsAssignedToCenter.data(),
            asserting_cast<int>(_pointToCenterAssignment.numPointsAssignedToCenter.size()), MPI_UINT64_T, MPI_SUM,
            MPI_COMM_WORLD);
        assert(_pointToCenterAssignment.numPointsAssignedToCenter.size() == numCenters());

        // Calculate new center positions
        for (size_t centerIdx = 0; centerIdx < numCenters(); centerIdx++) {
            for (size_t dimension = 0; dimension < numDimensions(); dimension++) {
                _centers[centerIdx + dimension] =
                    contribToCenterPosition[centerIdx + dimension]
                    / static_cast<data_t>(_pointToCenterAssignment.numPointsAssignedToCenter[centerIdx]);
            }
        }
    }

    void compute() {
        assert(_data.valid());

        // Pick initial centers, for now on rank 0
        pickInitialCenters();

        // The data is already distributed across the ranks

        // Perform numIterations k-means iterations
        for (uint64_t iteration = 0; iteration < _numIterations; iteration++) {
            // Assign the points to centers
            assignPointsToCenters();

            // Compute contribution of local data points to the positions of the new centers
            computeNewCenters();
        }
    }

    void operator()() {
        compute();
    }

    private:
    kMeansData<data_t>      _data;
    kMeansData<data_t>      _centers; // Will be empty after construction, before the initial centers are picked.
    PointToCenterAssignment _pointToCenterAssignment;
    uint64_t                _numIterations;
    uint64_t                _numCenters;
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