#!/bin/env python3

import argparse
from math import floor, ceil, log2
import math
from functools import lru_cache
import sys
import cProfile


class BinomialCoefficientCacheFixedN:
    def __init__(self, maxn):
        assert(isinstance(maxn, int))
        assert(maxn > 0)
        self._n = maxn

        # The binomial coefficient is symmetric, i.e. comb(n, k) = comb(n, n - k). Therefore, we can always
        # compute comb(n, min(k, n - k)) to save some time.
        maxk = int(ceil(maxn / 2))

        self._C = [1]
        numerator = 1
        for i in range(self._n, 0, -1):
            numerator *= i
            denominator = (self._n - i + 1)
            assert(numerator % denominator == 0)
            numerator //= denominator
            self._C.append(numerator)

    # Yes, this layer of cacheing actually results in a speedup.
    @lru_cache(maxsize=1048576)
    def __call__(self, n, k):
        # See __init__ for explanation of symmetry.
        assert(self._n == n)
        k = min(k, n - k)
        return self._C[k]

    _C = []
    _n = 0

# def get_failure_probability_for_fixed_f(p, r, f, g_choose_j):
#     assert(isinstance(p, int))
#     assert(isinstance(r, int))
#     assert(p % r == 0)
#     g = p // r
#
#     sign = 1
#     prob = 0
#     running_p = p - r
#     running_f = f - r
#     p_choose_f = math.comb(p - r, f - r)
#     # Idee: Iterationsrichtung umdrehen, dadurch das initiale math.comb sparen. Eine weitere Iteration -> math.comb unten sparen
#     for k in range(1, int(floor(f / r)) + 1):
#         # prob += sign * math.comb(g, k) * math.comb(p - k * r, f - k * r)
#         # can we replace these bigint multiplications and divisions my additions (Pascal's triangle)?
#         while running_f > f - k * r:
#             assert(p_choose_f * running_f % running_p == 0)
#             p_choose_f = p_choose_f * running_f // running_p
#             running_f -= 1
#             running_p -= 1
#         assert(running_p == p - k * r)
#         assert(running_f == f - k * r)
#         assert(p_choose_f == math.comb(running_p, running_f))
#
#         assert(g_choose_j(g, k) == math.comb(g, k))
#         prob += sign * g_choose_j(g, k) * p_choose_f
#         sign *= -1
#
#     prob /= math.comb(p, f)
#     return prob


def per_failure_probability(p, r):
    assert(isinstance(p, int))
    assert(isinstance(r, int))
    assert(p > 0)
    assert(r > 0)
    assert(p % r == 0)

    # Compute the number of groups g and precompute C(g, k) for this g.
    g = p // r
    g_choose_j = BinomialCoefficientCacheFixedN(g)

    # Initialize the probabilties for each failure count f to the additive identity.
    prob_until_f = [0 for f in range(0, p + 1)]

    # The binomial coefficient is symmetric, i.e. comb(n, k) = comb(n, n - k). Therefore, we can
    # always compute comb(n, min(k, n - k)) to save some time.
    # This is currently not implemented.
    maxn = p
    maxk = maxn

    Cn = [0 for k in range(maxk + 1)]

    # Cache C[n - 1][k - 1] which will get overwritten one iteration before it is needed.
    Cnk_1 = 1
    Cnk_2 = 1

    # Iterate over all values for n and k (k <= n/2 because of symmetry)
    for n in range(0, maxn + 1):
        # for k in range(0, min(n, maxk) + 1):
        for k in range(0, maxk + 1):
            Cnk_2 = Cnk_1  # Used this round
            Cnk_1 = Cn[k]  # Used next round

            # Calculate the current value for comb(n, k)
            if k == 0 or k == maxn:
                Cn[k] = 1
            else:
                Cn[k] = Cnk_2 + Cn[k]

            # For which failure count f do we need the current coefficient?
            # TODO: Exploit the symmetry of the binomial coefficient.
            if (p - n) % r != 0:
                continue
            j = (p - n) // r
            f = k + p - n
            if 0 < f <= p and n < p and 1 <= j <= g:
                sign = -1 if (j + 1) % 2 == 1 else 1
                assert(g_choose_j(g, j) == math.comb(g, j))
                prob_until_f[f] += sign * g_choose_j(g, j) * Cn[k]
            elif j == 0:
                # Divide each value by p choose f to get a probability.
                prob_until_f[f] /= Cn[k]

    # Calculate the probability for f == from f <= and return.
    prob_exactly_f = [0.0] + [
        prob_until_f[i + 1] - prob_until_f[i]
        for i
        in range(0, len(prob_until_f) - 1)
    ]
    return prob_exactly_f


def exp_num_failures_until_idl(p, r):
    probs = per_failure_probability(p, r)
    return sum([
        f * prob
        for (f, prob)
        in zip(
            range(0, len(probs)),
            probs
        )])


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Calculate the probability of irrecoverable data loss using a theoretical formula.")
    parser.add_argument("-p", "--processors",
                        help="logarithm (base 2) of max. number of processors (PEs) to use", type=int)
    parser.add_argument("-r", "--replicas",
                        help="logarithm (base 2) of max. number of replicas to use for each block range", type=int, default=3)

    cProfile.run('exp_num_failures_until_idl(2048, 4)')

    args = parser.parse_args()
    max_logp = int(args.processors)
    max_p = 2 ** max_logp
    max_r = int(args.replicas)
    # rs = [1, 2, 4]
    rs = [4]
    max_r = 4

    # Print CSV header
    print("numberOfPEs,replicationLevel,roundsUntilIrrecoverableDataLoss")

    # Compute the failure probabilities
    first_p = int(log2(max_r))
    for r in rs:
        for logp in range(first_p, max_logp + 1):
            p = pow(2, logp)
            # print("========================")
            # print(f"r: {r}, p: {p}")
            print(f"{p},{r},{exp_num_failures_until_idl(p, r)}")
