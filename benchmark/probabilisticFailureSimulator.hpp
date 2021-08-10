#include <random>
#include <unordered_set>

#include "restore/helpers.hpp"

class ProbabilisticFailureSimulator {
    public:
    ProbabilisticFailureSimulator(const unsigned long seed, const double failureProbability)
        : gen(seed),
          dist(failureProbability),
          prob(failureProbability) {}

    void changeFailureProbability(const double probability) {
        dist = std::geometric_distribution<>(probability);
        prob = probability;
    }

    void
    getFailingRanks(int numRanks, std::unordered_set<int>& outVec, bool skipFirstRank = false, int maxFailures = -1) {
        if (maxFailures == -1) {
            maxFailures = numRanks;
        }

        int pos = 0;
        if (skipFirstRank) {
            pos = 1;
        }

        while (pos < numRanks && outVec.size() < asserting_cast<size_t>(maxFailures)) {
            pos += dist(gen);
            if (pos < numRanks) {
                outVec.insert(pos);
            }
            pos++;
        }
    }

    private:
    std::mt19937                  gen;
    std::geometric_distribution<> dist;
    double                        prob;
};
