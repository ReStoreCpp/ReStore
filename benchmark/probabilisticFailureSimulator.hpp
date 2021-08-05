#include <random>
#include <unordered_set>

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

    void getFailingRanks(const int numRanks, std::unordered_set<int>& outVec) {
        if (prob <= 0)
            return;
        int pos = 0;
        while (pos < numRanks) {
            pos += dist(gen);
            if (pos < numRanks)
                outVec.insert(pos);
            pos++;
        }
    }

    private:
    std::mt19937                  gen;
    std::geometric_distribution<> dist;
    double                        prob;
};
