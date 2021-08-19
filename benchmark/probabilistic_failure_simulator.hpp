#include <algorithm>
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

    // Fails every rank with the given failureProbability
    void
    maybeFailRanks(int numRanks, std::unordered_set<int>& outVec) {
        assert(prob >= 0.0);
        assert(prob <= 1.0);

        int pos = 0;
        while (pos < numRanks && outVec.size()) {
            pos += dist(gen);
            if (pos < numRanks) {
                outVec.insert(pos);
            }
            pos++;
        }
    }

    // Randomly draws numberOfFailures ranks from [0, numRanks).
    void failRanksNow(
        int numRanks, unsigned int numberOfFailures, std::unordered_set<int>& outVec, bool skipFirstRank = false) {
        // We have to draw numberOfFailures elements from [0,numRanks). For this, we generate the consecutive sequence
        // [0,numRanks), shuffle it and return the first numberOfFailures elements.
        int startWith = 0;
        if (skipFirstRank) {
            startWith = 1;
        }
        std::vector<int> ranksToFail(throwing_cast<size_t>(numRanks));
        std::iota(begin(ranksToFail), end(ranksToFail), startWith);

        std::shuffle(begin(ranksToFail), end(ranksToFail), gen);

        // Copy over the first numberOfFailures elements to outVec.
        outVec.insert(begin(ranksToFail), begin(ranksToFail) + numberOfFailures);
    }

    private:
    std::mt19937                  gen;
    std::geometric_distribution<> dist;
    double                        prob;
};