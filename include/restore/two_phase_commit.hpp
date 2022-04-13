#include <optional>
#include <type_traits>
#include <utility>

template <class T>
class TwoPhaseCommit {
    static_assert(
        std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>,
        "T must be copy constructible and copy assignable.");

    public:
    template <class... Args>
    TwoPhaseCommit(Args&&... args) : _activeCopy(std::forward<Args>(args)...),
                                     _backupCopy(_activeCopy) {}

    TwoPhaseCommit() : _activeCopy(std::nullopt), _backupCopy(std::nullopt) {}

    TwoPhaseCommit(const TwoPhaseCommit&) = delete;
    TwoPhaseCommit& operator=(const TwoPhaseCommit&) = delete;
    TwoPhaseCommit(TwoPhaseCommit&&)                 = delete;
    TwoPhaseCommit& operator=(TwoPhaseCommit&&) = delete;

    template <class... Args>
    void emplace(Args&&... args) {
        _activeCopy = T(std::forward<Args>(args)...);
        _backupCopy = _activeCopy;
    }

    T& operator*() {
        _assertEngaged();
        return activeCopy();
    }

    const T& operator*() const {
        _assertEngaged();
        return activeCopy();
    }

    T* operator->() {
        _assertEngaged();
        return &(activeCopy());
    }

    const T* operator->() const {
        _assertEngaged();
        return &(activeCopy());
    }

    bool hasValue() const {
        return _activeCopy.has_value();
    }

    const T& activeCopy() const {
        _assertEngaged();
        return _activeCopy.value();
    }

    T& activeCopy() {
        _assertEngaged();
        return _activeCopy.value();
    }

    const T& backupCopy() const {
        _assertEngaged();
        return _backupCopy.value();
    }

    void commit() {
        _assertEngaged();
        _backupCopy = _activeCopy;
    }

    void rollback() {
        _assertEngaged();
        _activeCopy = _backupCopy;
    }

    private:
    void _assertEngaged() const {
        if (!_activeCopy) {
            throw std::runtime_error("There is no object stored yet.");
        }
        assert(_backupCopy);
    }

    std::optional<T> _activeCopy;
    std::optional<T> _backupCopy;
};