#ifndef RESTORE_HELPERS_H
#define RESTORE_HELPERS_H

#include <cstdint>
#include <limits>
#include <type_traits>

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
inline constexpr uint8_t operator""_byte(unsigned long long arg) noexcept {
    return static_cast<uint8_t>(arg);
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
template <class To>
struct in_range {
    template <class From>
    static constexpr bool check(From value) noexcept {
        static_assert(std::is_integral_v<From>, "From has to be an integral type.");
        static_assert(std::is_integral_v<To>, "To has to be an integral type.");

        static_assert(!std::is_unsigned_v<From> || std::numeric_limits<From>::min() == 0);
        static_assert(!std::is_unsigned_v<To> || std::numeric_limits<To>::min() == 0);

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
};

#endif // Include guard