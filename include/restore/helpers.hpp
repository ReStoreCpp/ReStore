#ifndef RESTORE_HELPERS_H
#define RESTORE_HELPERS_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <type_traits>
#include <complex>

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
// it. Depeneding on which version we choose, this check is either an assert or throws an exception.
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
    }

    assert(mpi_type != MPI_DATATYPE_NULL);

    return mpi_type;
}

#endif // Include guard
