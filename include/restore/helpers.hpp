#ifndef RESTORE_HELPERS_H
#define RESTORE_HELPERS_H

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <type_traits>

// TODO: Write UnitTests

// Suppress compiler warnings about unused variables
#define UNUSED(expr) (void)(expr)

#ifdef BACKWARD_CXX11
// Use the backward-cpp library to print a pretty backtrace
void print_stacktrace() {
    backward::StackTrace stacktrace;
    backward::Printer    printer;
    stacktrace.load_here(32);
    printer.print(stacktrace);
}
#endif

// Define some usefull user-defined literals
inline constexpr std::byte operator""_byte(unsigned long long arg) noexcept {
    return static_cast<std::byte>(arg);
}

inline constexpr uint8_t operator""_uint8(unsigned long long arg) noexcept {
    return static_cast<uint8_t>(arg);
}

inline constexpr uint16_t operator""_uint16(unsigned long long arg) noexcept {
    return static_cast<uint16_t>(arg);
}

inline constexpr uint32_t operator""_uint32(unsigned long long arg) noexcept {
    return static_cast<uint32_t>(arg);
}

inline constexpr uint64_t operator""_uint64(unsigned long long arg) noexcept {
    return static_cast<uint64_t>(arg);
}

// Default debug code to not be included
#ifndef NDEBUG
    #define DEBUG
#endif

// in_range<To>(From value) checks if value can be safely casted into type To.
template <class To, class From>
constexpr bool in_range(From value) noexcept {
    static_assert(std::is_integral_v<From>, "From has to be an integral type.");
    static_assert(std::is_integral_v<To>, "To has to be an integral type.");

    static_assert(!std::is_unsigned_v<From> || std::numeric_limits<From>::min() == 0);
    static_assert(!std::is_unsigned_v<To> || std::numeric_limits<To>::min() == 0);

    static_assert(std::numeric_limits<From>::digits <= 64);
    static_assert(std::numeric_limits<To>::digits <= 64);

    if constexpr (std::is_unsigned_v<From> && std::is_unsigned_v<To>) {
        return static_cast<uintmax_t>(value) <= static_cast<uintmax_t>(std::numeric_limits<To>::max());
    } else if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        return static_cast<intmax_t>(value) >= static_cast<intmax_t>(std::numeric_limits<To>::min())
               && static_cast<intmax_t>(value) <= static_cast<intmax_t>(std::numeric_limits<To>::max());
    } else if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
        if (value < 0) {
            return false;
        } else {
            return static_cast<uintmax_t>(value) <= static_cast<uintmax_t>(std::numeric_limits<To>::max());
        }
    } else if constexpr (std::is_unsigned_v<From> && std::is_signed_v<To>) {
        return static_cast<uintmax_t>(value) <= static_cast<uintmax_t>(std::numeric_limits<To>::max());
    }
}

// The following two functions use the above in_range to check if a value can be safely casted and if so, static_cast
// it. Depending on which version we choose, this check is either an assert or throws an exception.
template <class To, class From>
constexpr To asserting_cast(From value) noexcept {
    assert(in_range<To>(value));
    return static_cast<To>(value);
}

template <class To, class From>
constexpr To throwing_cast(From value) {
    if (!in_range<To>(value)) {
        throw std::range_error("string(value) is not not representable the target type");
    } else {
        return static_cast<To>(value);
    }
}

template <size_t N>
MPI_Datatype mpi_custom_continuous_type() {
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(N, MPI_CHAR, &type);
        MPI_Type_commit(&type);
    }
    return type;
}

// Translate template parameter T to an MPI_Datatype
// Based on https://gist.github.com/2b-t/50d85115db8b12ed263f8231abf07fa2
template <typename T>
[[nodiscard]] constexpr MPI_Datatype get_mpi_type() noexcept {
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;

    if constexpr (std::is_same_v<T, char>) {
        mpi_type = MPI_CHAR;
    } else if constexpr (std::is_same_v<T, signed char>) {
        mpi_type = MPI_SIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        mpi_type = MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, wchar_t>) {
        mpi_type = MPI_WCHAR;
    } else if constexpr (std::is_same_v<T, signed short>) {
        mpi_type = MPI_SHORT;
    } else if constexpr (std::is_same_v<T, unsigned short>) {
        mpi_type = MPI_UNSIGNED_SHORT;
    } else if constexpr (std::is_same_v<T, signed int>) {
        mpi_type = MPI_INT;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        mpi_type = MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T, signed long int>) {
        mpi_type = MPI_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long int>) {
        mpi_type = MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<T, signed long long int>) {
        mpi_type = MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long long int>) {
        mpi_type = MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<T, float>) {
        mpi_type = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        mpi_type = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, long double>) {
        mpi_type = MPI_LONG_DOUBLE;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        mpi_type = MPI_INT8_T;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        mpi_type = MPI_INT16_T;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        mpi_type = MPI_INT32_T;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        mpi_type = MPI_INT64_T;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        mpi_type = MPI_UINT8_T;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        mpi_type = MPI_UINT16_T;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        mpi_type = MPI_UINT32_T;
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        mpi_type = MPI_UINT64_T;
    } else if constexpr (std::is_same_v<T, bool>) {
        mpi_type = MPI_C_BOOL;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        mpi_type = MPI_C_COMPLEX;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    } else if constexpr (std::is_same_v<T, std::complex<long double>>) {
        mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
    } else {
        mpi_type = mpi_custom_continuous_type<sizeof(T)>();
    }

    assert(mpi_type != MPI_DATATYPE_NULL);

    return mpi_type;
}

// Returns the identity for the requested MPI operation (MPI_Op). MINLOC und MAXLOC are undefined.
template <class data_t>
[[nodiscard]] constexpr data_t mpi_op_identity(MPI_Op op) {
    // We cannot use a switch, as MPI_OP is not a integer type.
    if (op == MPI_MAX) {
        return std::numeric_limits<data_t>::lowest();
    } else if (op == MPI_MIN) {
        return std::numeric_limits<data_t>::max();
    } else if (op == MPI_SUM) {
        return 0;
    } else if (op == MPI_PROD) {
        return 1;
    } else if (op == MPI_LAND) {
        return true;
    } else if (op == MPI_LOR) {
        return false;
    } else if (op == MPI_BAND) {
        return static_cast<data_t>(-1);
    } else if (op == MPI_BOR) {
        return 0;
    } else if (MPI_MAXLOC || op == MPI_MINLOC) {
        throw std::runtime_error("MPI_MAXLOC and MPI_MINLOC are not implemented");
    } else {
        throw std::runtime_error("Unknown MPI_Op");
    }
}

// Some bit twiddling helper functions.
template <typename Data>
constexpr uint8_t num_bits() {
    static_assert(std::is_pod_v<Data>, "Data has to be a POD.");
    return sizeof(Data) * 8;
}

template <typename Data>
inline int64_t bits_left_half(Data bytes) {
    static_assert(std::is_pod_v<Data>, "Data has to be a POD.");
    return bytes >> num_bits<Data>() / 2;
}

template <typename Data>
inline int64_t bits_right_half(Data bytes) {
    static_assert(std::is_pod_v<Data>, "Data has to be a POD.");
    return bytes & static_cast<Data>(-1) >> num_bits<Data>() / 2;
}

template <typename Data>
inline int64_t bits_combine_halves(Data left, Data right) {
    static_assert(std::is_pod_v<Data>, "Data has to be a POD.");
    return (left << num_bits<Data> / 2) | right;
}

// Searches the source operand for the most significant set bit (1 bit). If a most significant 1 bit is found, its bit
// index is stored in the destination operand. The source operand can be a register or a memory location; the
// destination operand is a register. The bit index is an unsigned offset from bit 0 of the source operand. If the
// content source operand is 0, the content of the destination operand is undefined.
template <typename Data>
inline uint8_t most_significant_bit_set(Data bytes) {
    static_assert(std::is_pod_v<Data>, "Data has to be a POD.");

    if (bytes == 0) {
        return 0;
    }

    Data msb;
    asm("bsr %1,%0" : "=r"(msb) : "r"(bytes));
    return asserting_cast<uint8_t>(msb);
}

// Simple and fast string hash function taken from http://www.cse.yorku.ca/~oz/hash.html
uint32_t hash_djb2(const unsigned char* str) {
    uint32_t hash = 5381;
    uint32_t c;

    while ((c = *str++) != 0) {
        //  hash * 33 + c
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

uint32_t hash_djb2(const char* str) {
    return hash_djb2(reinterpret_cast<const unsigned char*>(str));
}

uint32_t hash_djb2(const std::string& str) {
    return hash_djb2(str.c_str());
}

class ResultsCSVPrinter {
    public:
    ResultsCSVPrinter(std::ostream& stream, bool printHeader)
        : _stream(stream),
          _printHeaderWithNextResult(printHeader) {}

    template <class ValueType>
    void allResults(const std::string& key, const ValueType& value) {
        if (_startedPrinting) {
            throw std::runtime_error(
                "I already started printing, please register attributes concerning all results beforehand.");
        }
        _attributesForAllResults.emplace_back(key, _toString(value));
    }

    template <class ValueType>
    void thisResult(const std::string& key, const ValueType& value) {
        auto result = std::find_if(_attributesPerResult.begin(), _attributesPerResult.end(), [&key](const kvPair& kv) {
            return key == kv.key;
        });

        if (result == _attributesPerResult.end()) {
            if (_startedPrinting) {
                throw std::runtime_error("I already started printing and you gave me a new attribute. All results must "
                                         "have exactly the same attributes.");
            } else {
                _attributesPerResult.emplace_back(key, _toString(value));
            }
        } else {
            result->value            = _toString(value);
            result->setForThisResult = true;
        }
    }

    template <class ValueType>
    void thisResult(const char* key, const std::vector<ValueType>& value) {
        thisResult(std::string(key), value);
    }

    template <class ValueType>
    void thisResult(const std::vector<std::pair<std::string, ValueType>>& kvVector) {
        for (const auto& kv: kvVector) {
            thisResult(kv.first, kv.second);
        }
    }

    template <class ValueType>
    void thisResult(const std::vector<std::pair<const char*, ValueType>>& kvVector) {
        for (const auto& kv: kvVector) {
            thisResult(kv.first, kv.second);
        }
    }

    void finalizeAndPrintResult() {
        _checkAllAttributesSetForThisResult();
        _startedPrinting = true;

        // Print the CSV header
        if (_printHeaderWithNextResult) {
            _printHeaderWithNextResult = false;
            _printHeader();
        }

        // Print the CSV row for one result
        _printResult();

        // Reset the attributes for the next result
        std::for_each(
            _attributesPerResult.begin(), _attributesPerResult.end(), [](kvPair& kv) { kv.setForThisResult = false; });
    }

    private:
    struct kvPair {
        kvPair(const std::string& _key, const std::string& _value) : key(_key), value(_value), setForThisResult(true) {}
        std::string key;
        std::string value;
        bool        setForThisResult; // Indicated if this attribute has been written for this result, ignored for
                                      // attributes which are set for all results. not set for all results.
    };

    template <class ValueType>
    std::string _toString(const ValueType& value) const {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    void _printHeader() {
        _printRow([](const kvPair& kv) { return kv.key; });
    }

    void _printResult() {
        _printRow([](const kvPair& kv) { return kv.value; });
    }

    template <class F>
    void _printRow(F selector) {
        // Print the attributes identical for all results
        bool firstAttribute = true;
        for (auto& kv: _attributesForAllResults) {
            if (!firstAttribute) {
                _stream << ",";
            } else {
                firstAttribute = false;
            }
            _stream << selector(kv);
        }

        // Print the attributes different for each result
        for (auto& kv: _attributesPerResult) {
            _stream << "," << selector(kv);
        }

        // Print EOL
        _stream << std::endl;
    }

    void _checkAllAttributesSetForThisResult() const {
        auto notSet = std::find_if_not(_attributesPerResult.begin(), _attributesPerResult.end(), [](const kvPair& kv) {
            return kv.setForThisResult;
        });
        if (notSet != _attributesPerResult.end()) {
            throw std::runtime_error("Attribute " + notSet->key + " is not set for this result.");
        }
    }

    std::ostream&       _stream;
    std::vector<kvPair> _attributesForAllResults;
    std::vector<kvPair> _attributesPerResult;

    bool _printHeaderWithNextResult;
    bool _startedPrinting = false;
};

#endif // Include guard
