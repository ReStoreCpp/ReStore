#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "restore/helpers.hpp"

#include "xxhash.h"

class LCGPseudoRandomPermutation {
    public:
    LCGPseudoRandomPermutation(int64_t max_value) : _max_value(max_value) {
        _choose_modulo(_max_value);
        _choose_a();
    }

    int64_t f(int64_t n) const {
        // We use cycle walking to ensure, that the generated number is at most _max_value
        do {
            n = _mod(n * _a + _c);
        } while (n > _max_value);
        return n;
    }

    int64_t finv(int64_t n) const {
        do {
            n = _mod((n - _c) * _ainv);
        } while (n > _max_value);
        return n;
    }

    private:
    int64_t _max_value;
    int64_t _modulo;
    int64_t _modulo_and_mask;
    int64_t _c =
        1; // Satisfies the Hull-Dobell theorem. The distribution of random numbers is not sensitive to the value of c.
    int64_t _a    = 0;
    int64_t _ainv = 0;

    void _choose_modulo(int64_t max_value) {
        // See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
        // Round max_value up to the next highest power of 2.
        max_value--; // In case max_value already is a power of two, we do not want to change it.
        // Copy the highest set bit to all the lower bits.
        max_value |= max_value >> 1;
        max_value |= max_value >> 2;
        max_value |= max_value >> 4;
        max_value |= max_value >> 8;
        max_value |= max_value >> 16;
        max_value |= max_value >> 32;

        _modulo_and_mask = max_value;     // Set all bits except the highest one of the bitmask.
        _modulo          = max_value + 1; // Add 1, which will carry over to the first 0 after all the ones.
        _modulo_and_mask |= _modulo;      // Set the highest bit of the bitmask.
    }

    // TODO: How do we choose a proper a?
    // Satisfy Hull-Dobell theorem
    void _choose_a() {
        // a has to be < m
        // a should also be congruent to 5 mod 8; i.e. be divisible by 4 but not by 8.
        // A m is a power of two, this means a and m only have the common divisors 1, 2, and 4.
        _a    = 5; // TODO Change this!!
        _ainv = _modulo_multiplicative_inverse(_a, _modulo);
    }

    int64_t _mod(int64_t n) const {
        return n & _modulo_and_mask;
    }

    int64_t _modulo_multiplicative_inverse(int64_t a, int64_t m) const {
        // https://rosettacode.org/wiki/Modular_inverse#C.2B.2B
        int64_t m0 = m, t, q;
        int64_t x0 = 0, x1 = 1;

        if (m == 1) {
            return 1;
        }

        while (a > 1) {
            q = a / m;
            t = m, m = a % m, a = t;
            t = x0, x0 = x1 - q * x0, x1 = t;
        }

        if (x1 < 0) {
            x1 += m0;
        }

        return x1;
    }
};

class FeistelPseudoRandomPermutation {
    public:
    static FeistelPseudoRandomPermutation buildPermutation(uint64_t maxValue, uint64_t seed) {
        std::mt19937_64 rng(seed);
        return buildPermutation(maxValue, rng);
    }

    // //! Each rank will get a different permutation!
    // static FeistelPseudoRandomPermutation buildPermutation(uint64_t maxValue) {
    //     std::random_device rd;
    //     return buildPermutation(maxValue, rd);
    // }

    template <typename EntropySource>
    static FeistelPseudoRandomPermutation buildPermutation(uint64_t maxValue, EntropySource& entropySource) {
        const uint8_t num_rounds = 4;

        std::vector<XXH64_hash_t> keys;
        keys.reserve(num_rounds);
        for (uint8_t round = 0; round < num_rounds; round++) {
            keys.push_back(entropySource());
        }

        return FeistelPseudoRandomPermutation(maxValue, keys);
    }

    FeistelPseudoRandomPermutation(
        uint64_t max_value, const std::vector<XXH64_hash_t>& keys, const uint8_t num_rounds = 4)
        : _max_value(max_value),
          _num_rounds(num_rounds),
          _keys(keys) {
        if (_num_rounds != keys.size()) {
            throw std::invalid_argument("Number of keys must be equal to the number of rounds.");
        }
        _compute_size_of_halves(_max_value);
    }

    uint64_t f(uint64_t n) const {
        // We use cycle walking to ensure, that the generated number is at most _max_value
        return _cycle_walk(n, false);
    }

    uint64_t finv(uint64_t n) const {
        return _cycle_walk(n, true);
    }

    private:
    uint64_t                  _max_value;
    uint8_t                   _num_rounds;
    uint8_t                   _bits_left_half;
    uint8_t                   _bits_right_half;
    uint8_t                   _left_half_shift_width;
    uint8_t                   _extra_bits;
    uint64_t                  _right_half_mask;
    std::vector<XXH64_hash_t> _keys;

    uint64_t _cycle_walk(uint64_t n, bool reverse) const {
        if (n > _max_value) {
            throw std::invalid_argument("n cannot be larger than the maximum value given to the constructor.");
        }

        do {
            n = _feistel(n, reverse);
        } while (n > _max_value);
        return n;
    }

    void _compute_size_of_halves(uint64_t max_value) {
        // Compute the sizes of the right and left halves of the permutation.
        int num_significant_bits = most_significant_bit_set(max_value) + 1;

        // An uneven number of bits would require a more difficult to implement unbalanced Feistel network. We sacrifice
        // some speed (expected factor 2 more cycle walks) for reduced code complexity.
        if (num_significant_bits % 2 == 1) {
            num_significant_bits++;
        }

        // TODO asserting casts
        _bits_left_half  = asserting_cast<uint8_t>(num_significant_bits / 2);
        _bits_right_half = asserting_cast<uint8_t>(num_significant_bits - _bits_left_half);

        // Precompute the bitmasks and shift constants for the _left_half() and _right_half() calls.
        _extra_bits            = asserting_cast<uint8_t>(num_bits<uint64_t>() - _bits_left_half - _bits_right_half);
        _left_half_shift_width = asserting_cast<uint8_t>(num_bits<uint64_t>() - _bits_right_half);
        _right_half_mask       = -1ull;
        _right_half_mask >>= num_bits<uint64_t>() - _bits_right_half;
    }

    void _choose_keys() {
        std::random_device rd;
        _keys.reserve(_num_rounds);
        for (uint8_t round = 0; round < _num_rounds; round++) {
            _keys.push_back(rd());
        }
    }

    uint64_t _left_half(uint64_t n) const {
        assert(_bits_left_half > 0);
        assert(_bits_left_half < num_bits<uint64_t>());

        // First, cut off the part left of the left half (as the number of bits in the storage data type might be
        // greater than the the number of bits we are looking at.)
        auto left_half = n << _extra_bits;

        // Next, shift the left half to the right. We also undo the lift shift from the previous step here.
        left_half >>= _left_half_shift_width;

        assert(most_significant_bit_set(left_half) <= _bits_left_half);
        return left_half;
    }

    uint64_t _right_half(uint64_t n) const {
        assert(_bits_right_half > 0);
        assert(_bits_right_half < num_bits<uint64_t>());

        const auto right_half = n & _right_half_mask;

        assert(most_significant_bit_set(right_half) <= _bits_right_half);
        return right_half;
    }

    uint64_t _combine_halves(uint64_t left, uint64_t right) const {
        assert(most_significant_bit_set(left) <= _bits_left_half);
        assert(most_significant_bit_set(right) <= _bits_right_half);
        return left << _bits_right_half | right;
    }

    uint64_t _feistel(uint64_t n, bool reverse = false) const {
        // We might get a number which is greater than max_value when cycle walking. In the case max_value's MSB is at
        // an uneven position, we could even get a value which's MSB is at a lefter position than max_values's. This is
        // due to our simplification of using an extra bit to avoid an unbalanced Feistel network. We therefore can only
        // check if the MSB of the given parameter is lefter than the number of bits in our Feistel network.
        // assert(_bits_left_half + _bits_right_half >= most_significant_bit_set(n));

        // Split the input in two halves.
        uint64_t left  = _left_half(n);
        uint64_t right = _right_half(n);

        // assert(most_significant_bit_set(left) <= _bits_left_half);
        // assert(most_significant_bit_set(right) <= _bits_right_half);
        // assert(most_significant_bit_set(_combine_halves(left, right)) <= _bits_left_half + _bits_right_half);
        // assert(_combine_halves(left, right) == n);

        // If we want to compute the reverse permutation, we need to reverse the order of the keys.
        if (!reverse) {
            // For each key ...
            for (int16_t keyIdx = 0; keyIdx < asserting_cast<int16_t>(_keys.size()); keyIdx++) {
                // .. apply the Feistel function.
                auto&    key = _keys[asserting_cast<size_t>(keyIdx)];
                uint64_t tmp = left ^ _right_half(_xxhash(right, key));
                left         = right;
                right        = tmp;

                assert(most_significant_bit_set(left) <= _bits_left_half);
                assert(most_significant_bit_set(right) <= _bits_right_half);
                assert(most_significant_bit_set(_combine_halves(left, right)) <= _bits_left_half + _bits_right_half);
            }
        } else {
            // For each key ...
            for (int16_t keyIdx = asserting_cast<int16_t>(_keys.size() - 1); keyIdx >= 0; keyIdx--) {
                // .. apply the Feistel function.
                auto&    key = _keys[asserting_cast<size_t>(keyIdx)];
                uint64_t tmp = right ^ _right_half(_xxhash(left, key));
                right        = left;
                left         = tmp;

                assert(most_significant_bit_set(left) <= _bits_left_half);
                assert(most_significant_bit_set(right) <= _bits_right_half);
                assert(most_significant_bit_set(_combine_halves(left, right)) <= _bits_left_half + _bits_right_half);
            }
        }

        return _combine_halves(left, right);
    }

    template <class Data>
    inline XXH64_hash_t _xxhash(Data n, XXH64_hash_t key) const {
        static_assert(std::is_pod_v<Data>, "Data has to be a POD.");
        return XXH64(&n, sizeof(Data), key);
    }
};

class IdentityPermutation {
    public:
    static IdentityPermutation buildPermutation(uint64_t maxValueDummy = 0, uint64_t seedDummy = 0) {
        UNUSED(maxValueDummy);
        UNUSED(seedDummy);
        return IdentityPermutation();
    }

    uint64_t f(uint64_t n) const {
        return n;
    }

    uint64_t finv(uint64_t n) const {
        return n;
    }
};

template <typename Permutation = FeistelPseudoRandomPermutation>
class RangePermutation {
    public:
    RangePermutation(uint64_t maxValue, uint64_t lengthOfRanges, uint64_t seed)
        : _maxValue(maxValue),
          _lengthOfRanges(lengthOfRanges),
          _numRanges(RangePermutation::_computeNumRanges(maxValue, lengthOfRanges)),
          _permutation(Permutation::buildPermutation(_numRanges - 1, seed)) {
        assert(maxValue > 0);
        assert(lengthOfRanges > 0);
        assert(numRanges() > 0);
        assert(numRanges() <= maxValue + 1);
        assert(maxValue + 1 >= lengthOfRanges);
        // The last range might be smaller than the others.
        assert(numRanges() * lengthOfRanges >= maxValue);
    }

    inline uint64_t f(uint64_t n) const {
        assert(_range(n) < _numRanges);
        assert(_offset(n) < _lengthOfRanges);
        return _permutation.f(_range(n)) * _lengthOfRanges + _offset(n);
    }

    inline uint64_t finv(uint64_t n) const {
        assert(_range(n) < _numRanges);
        assert(_offset(n) < _lengthOfRanges);
        assert(_range(n) < _numRanges - 1 || _offset(n) < _elementsInLastRange());
        return _permutation.finv(_range(n)) * _lengthOfRanges + _offset(n);
    }

    inline uint64_t numRanges() const {
        return _numRanges;
    }

    inline uint64_t maxValue() const {
        return _maxValue;
    }

    inline uint64_t lengthOfRanges() const {
        return _lengthOfRanges;
    }

    // Given a block id, return the last id of the range \c blockId belongs to.
    // static struct idIsFromLeftSideOfBijection{} idIsFromLeftSideOfBijection;
    // static struct idIsFromRightSideOfBijection{} idIsFromRightSideOfBijection;

    inline uint64_t lastIdOfRange(uint64_t blockId) const {
        const auto rangeId = _range(blockId);
        if (rangeId == _numRanges - 1) {
            return _maxValue;
        } else {
            return rangeId * _lengthOfRanges + _lengthOfRanges - 1;
        }
    }

    private:
    const uint64_t    _maxValue;
    const uint64_t    _lengthOfRanges;
    const uint64_t    _numRanges;
    const Permutation _permutation;

    inline uint64_t _range(uint64_t n) const {
        return n / _lengthOfRanges;
    }

    inline uint64_t _offset(uint64_t n) const {
        return n % _lengthOfRanges;
    }

    inline static uint64_t _computeNumRanges(uint64_t maxValue, uint64_t lengthOfRanges) {
        assert(lengthOfRanges > 0);
        assert(maxValue > 0);
        assert(maxValue + 1 >= lengthOfRanges);
        if ((maxValue + 1) % lengthOfRanges == 0) {
            return (maxValue + 1) / lengthOfRanges;
        } else {
            return (maxValue + 1) / lengthOfRanges + 1;
        }
    }

    inline uint64_t _elementsInLastRange() const {
        if ((_maxValue + 1) % _lengthOfRanges == 0) {
            return _lengthOfRanges;
        } else {
            return (_maxValue + 1) % _lengthOfRanges;
        }
    }
};