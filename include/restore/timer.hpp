// *****************************************************************************************
// The Contents of this file are based on:
// tlx/multi_timer.hpp; Part of tlx - http://panthema.net/tlx by Timo Bingmann
//
// All rights reserved. This file is published under the Boost Software License, Version 1.0
// *****************************************************************************************
#ifndef RESTORE_TIMER_HPP
#define RESTORE_TIMER_HPP

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <ostream>
#include <utility>
#include <vector>

#include "helpers.hpp"

static std::mutex MultiTimer_add_mutex;

// MultiTimer can be used to measure time usage of different phases in a program
// or algorithm. It contains multiple named "timers", which can be activated
// without prior definition. At most one timer is running at any time, which
// means `start()` will stop any current timer and start a new one.
//
// Timers are identified by strings, which are passed as const char*, which MUST
// remain valid for the lifetime of the MultiTimer. Dynamic strings will not
// work, the standard way is to use plain string literals. The strings are hashed
// for faster searches.
//
// MultiTimer can also be used for multi-threading parallel programs. Each
// thread must create and keep its own MultiTimer instance, which can then be
// added together into a global MultiTimer object. The add() method of the
// global object is internally thread-safe using a global mutex.

// TODO Write UnitTests
class TimerRegister {
    public:
    // timer entry
    struct Entry {
        explicit Entry(const char* _name) noexcept
            : hash(hash_djb2(_name)),
              name(_name),
              duration(std::chrono::duration<double>::zero()){};

        // hash of name for faster search
        uint32_t hash;
        // reference to original string for comparison
        const char* name;
        // duration of this timer
        std::chrono::duration<double> duration;
    };

    // TimerRegister(const TimerRegister&) = default;
    // TimerRegister& operator=(const TimerRegister&) = default;
    // TimerRegister(TimerRegister&&)                 = default;
    // TimerRegister& operator=(TimerRegister&&) = default;

    //~TimerRegister() = default;

    // start new timer phase, stop the currently running one.
    void start(const char* timer) {
        if (strcmp(timer, "total") == 0) {
            throw std::invalid_argument("total is a reserved timer name and will be computed automatically.");
        }

        assert(timer != nullptr);
        uint32_t hash = hash_djb2(timer);
        if (running_ && hash == running_hash_ && strcmp(running_, timer) == 0) {
            static bool warning_shown = false;
            if (!warning_shown) {
                std::cout << "TimerRegister: trying to start timer " << timer << " twice!";
                std::cout << "TimerRegister: multi-threading is not supported, "
                          << "use .add()";
                warning_shown = true;
            }
        }
        stop();
        running_      = timer;
        running_hash_ = hash;
    }

    // stop the currently running timer.
    void stop() {
        auto new_time_point = std::chrono::high_resolution_clock::now();
        if (running_) {
            Entry& e = find_or_create(running_);
            e.duration += new_time_point - time_point_;
            total_duration_ += new_time_point - time_point_;
        }
        time_point_   = new_time_point;
        running_      = nullptr;
        running_hash_ = 0;
    }

    // zero timers.
    void reset() {
        timers_.clear();
        total_duration_ = std::chrono::duration<double>::zero();
    }

    // return name of currently running timer.
    const char* running() const {
        return running_;
    }

    // return timer duration in seconds of timer.
    double get(const char* name) {
        return find_or_create(name).duration.count();
    }

    // return total duration of all timers.
    double total() const {
        return total_duration_.count();
    }

    // print all timers as a TIMER line to os
    void print(const char* info, std::ostream& os) const {
        assert(!running_);

        os << "TIMER info=" << info;
        for (const Entry& timer: timers_) {
            os << ' ' << timer.name << '=' << timer.duration.count();
        }
        os << " total=" << total_duration_.count() << std::endl;
    }

    // print all timers as a TIMER line to stderr
    void print(const char* info) const {
        return print(info, std::cerr);
    }

    // add all timers from another, internally holds a global mutex lock,
    // because this is used to add thread values
    TimerRegister& add(const TimerRegister& b) {
        std::unique_lock<std::mutex> lock(MultiTimer_add_mutex);
        if (b.running_) {
            std::cout << "MultiTimer: trying to add running timer";
        }
        for (const Entry& t: b.timers_) {
            Entry& e = find_or_create(t.name);
            e.duration += t.duration;
        }
        total_duration_ += b.total_duration_;
        return *this;
    }

    // add all timers from another, internally holds a global mutex lock,
    // because this is used to add thread values
    TimerRegister& operator+=(const TimerRegister& b) {
        return add(b);
    }

    // Return the reference to this singleton object.
    static TimerRegister& instance() {
        return _instance;
    }

    // Stop the current timer and push it to the stack, start the new timer.
    // If no timer is currently running, the pop() corresponding to this push() will not start a timer.
    void push(const char* timer) {
        timerStack_.push_back(running_);
        start(timer);
    }

    // Stop the current timer; pop and start a timer from the stack.
    void pop() {
        if (timerStack_.empty()) {
            throw std::runtime_error("pop() on empty timer stack");
        }
        auto back = timerStack_.back();
        if (back != nullptr) {
            start(back);
        }
        timerStack_.pop_back();
    }

    // Returns a vector of (name, duration) pairs.
    std::vector<std::pair<const char*, double>> getAllTimers() {
        std::vector<std::pair<const char*, double>> result;
        std::transform(timers_.begin(), timers_.end(), std::back_inserter(result), [](const Entry& entry) {
            assert(entry.name != nullptr);
            assert(strcmp(entry.name, "total") != 0);
            return std::make_pair(entry.name, entry.duration.count());
        });
        result.push_back(std::make_pair("total", total()));
        return result;
    }

    private:
    static TimerRegister _instance;
    TimerRegister() : total_duration_(std::chrono::duration<double>::zero()), running_(nullptr), running_hash_(0) {}

    // array of timers
    std::vector<Entry> timers_;

    // stack of timers
    std::vector<const char*> timerStack_;

    // total duration
    std::chrono::duration<double> total_duration_;

    // currently running timer name
    const char* running_;
    // hash of running_
    uint32_t running_hash_;
    // start of currently running timer name
    std::chrono::time_point<std::chrono::high_resolution_clock> time_point_;

    // Internal method to find or create new timer entries.
    Entry& find_or_create(const char* name) {
        // Search the timer array for the given timer, use hashing to speed up the search.
        uint32_t hash = hash_djb2(name);
        for (size_t i = 0; i < timers_.size(); ++i) {
            if (timers_[i].hash == hash && strcmp(timers_[i].name, name) == 0)
                return timers_[i];
        }

        // The timer was not found, create and return a new entry.
        timers_.emplace_back(name);
        return timers_.back();
    }
};

TimerRegister TimerRegister::_instance;

// RAII Scoped MultiTimer switcher: switches the timer of a MultiTimer on
// construction and back to old one on destruction.
class ScopedMultiTimerSwitch {
    public:
    // construct and timer to switch to
    ScopedMultiTimerSwitch(const char* new_timer) : timer_(TimerRegister::instance()), previous_(timer_.running()) {
        if (new_timer != nullptr) { // This enables us to pause the currently running timer without starting an new one.
            timer_.start(new_timer);
        } else {
            timer_.stop();
        }
    }

    // change back timer to previous timer.
    ~ScopedMultiTimerSwitch() {
        if (previous_ != nullptr) { // If no timer was previously running we don't need to switch back.
            timer_.start(previous_);
        } else {
            timer_.stop();
        }
    }

    protected:
    // reference to MultiTimer
    TimerRegister& timer_;

    // previous timer, used to switch back to on destruction
    const char* previous_;
};

#ifdef ENABLE_TIMERS
    #define TIME_BLOCK(NAME)          ScopedMultiTimerSwitch(NAME)
    #define TIME_PAUSE_BLOCK()        ScopedMultiTimerSwitch(nullptr)
    #define TIME_NEXT_SECTION(NAME)   TimerRegister::instance().start(NAME)
    #define TIME_PUSH_AND_START(NAME) TimerRegister::instance().push(NAME)
    #define TIME_POP(NAME)            TimerRegister::instance().pop()
    #define TIME_STOP()               TimerRegister::instance().stop()
#else
    #define TIME_BLOCK(NAME)          ;
    #define TIME_PAUSE_BLOCK()        ;
    #define TIME_NEXT_SECTION(NAME)   ;
    #define TIME_PUSH_AND_START(NAME) ;
    #define TIME_POP(NAME)            ;
    #define TIME_STOP()               ;
#endif

/******************************************************************************/
#endif // RESTORE_TIMER_HPP