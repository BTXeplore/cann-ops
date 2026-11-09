#pragma once

// Host test doubles for the exact template kernel entry point. They model values,
// bounds, byte strides, and cross-thread visibility at a simulated barrier.
// AI Core pipe timing and hardware scheduling are outside this model.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __global__
#define __gm__
#define KERNEL_TASK_TYPE_DEFAULT(...)
using half = _Float16;
using GM_ADDR = uint8_t*;

namespace AscendC {
enum class QuePosition { VECIN, VECOUT };
enum class TPosition { VECCALC };
enum class HardEvent { MTE2_S, S_V, V_S };
enum class RoundMode { CAST_NONE };
constexpr int PIPE_V = 0;
constexpr int PIPE_ALL = 1;

struct WorkCounts {
    size_t reads = 0;
    size_t inputDma = 0;
    size_t vectorToScalar = 0;
    bool readOffDiagonal = false;
};
inline thread_local WorkCounts* counts = nullptr;
inline thread_local uint32_t blockIndex = 0;
inline uint32_t blockCount = 1;
inline std::barrier<>* blockBarrier = nullptr;

inline uint32_t GetBlockIdx() { return blockIndex; }
inline uint32_t GetBlockNum() { return blockCount; }
template <bool VectorOnly> void SyncAll() {
    static_assert(VectorOnly);
    if (!blockBarrier) throw std::runtime_error("SyncAll without a multi-core launch");
    blockBarrier->arrive_and_wait();
}

template <typename T> struct Matrix {
    T* identity = nullptr;
    size_t stride = 0, elements = 0;
    std::vector<T> diagonal;
    std::vector<uint32_t> visits;
};
template <typename T> inline Matrix<T> matrix;

template <typename T> class GlobalTensor {
public:
    void SetGlobalBuffer(T* data, size_t size = 0) { data_ = data; size_ = size; }
    GlobalTensor operator[](size_t i) const { auto copy = *this; copy.offset_ += i; return copy; }
    bool IsInput() const { return data_ == matrix<T>.identity; }
    T GetValue(size_t i) const {
        auto& m = matrix<T>;
        const size_t at = i + offset_;
        if (!data_ || (size_ && at >= size_)) throw std::out_of_range("GM view");
        if (!IsInput()) return data_[at];
        if (at >= m.elements) throw std::out_of_range("GM input");
        ++counts->reads;
        if (at % m.stride != 0 || at / m.stride >= m.diagonal.size()) {
            counts->readOffDiagonal = true;
            return static_cast<T>(std::numeric_limits<float>::quiet_NaN());
        }
        const auto index = at / m.stride;
        std::atomic_ref<uint32_t>(m.visits[index]).fetch_add(1, std::memory_order_relaxed);
        return m.diagonal[index];
    }
    void SetValue(size_t i, T v) const {
        if (!data_ || IsInput() || offset_ + i >= size_) throw std::out_of_range("GM output");
        data_[offset_ + i] = v;
    }
private:
    T* data_ = nullptr;
    size_t size_ = 0, offset_ = 0;
};

using Buffer = std::shared_ptr<std::vector<uint64_t>>;

template <typename T> class LocalTensor {
public:
    explicit LocalTensor(Buffer b, size_t offset = 0) : buf_(std::move(b)), offset_(offset) {}
    LocalTensor operator[](size_t i) const { return LocalTensor(buf_, offset_ + i * sizeof(T)); }
    T GetValue(size_t i) const { return at(i); }
    void SetValue(size_t i, T v) const { at(i) = v; }
private:
    T& at(size_t i) const {
        const size_t byte = offset_ + i * sizeof(T);
        if (byte + sizeof(T) > buf_->size() * sizeof(uint64_t)) throw std::out_of_range("UB");
        return *reinterpret_cast<T*>(reinterpret_cast<char*>(buf_->data()) + byte);
    }
    Buffer buf_;
    size_t offset_;
};

struct Storage {
    void init(size_t bytes) { buffer = std::make_shared<std::vector<uint64_t>>((bytes + 7) / 8); }
    Buffer buffer;
};
template <QuePosition, int> struct TQue : Storage {
    template <typename T> LocalTensor<T> AllocTensor() { return LocalTensor<T>(buffer); }
    template <typename T> LocalTensor<T> DeQue() { return LocalTensor<T>(buffer); }
    template <typename T> void EnQue(LocalTensor<T>) {}
    template <typename T> void FreeTensor(LocalTensor<T>) {}
};
template <TPosition> struct TBuf : Storage {
    template <typename T> LocalTensor<T> Get() { return LocalTensor<T>(buffer); }
};
struct TPipe {
    void InitBuffer(Storage& b, size_t bytes) { b.init(bytes); }
    void InitBuffer(Storage& b, int slots, size_t bytes) { b.init(slots * bytes); }
    int FetchEventID(HardEvent) { return 0; }
};
template <HardEvent Event> void SetFlag(int) {
    if constexpr (Event == HardEvent::V_S) ++counts->vectorToScalar;
}
template <HardEvent> void WaitFlag(int) {}
template <int> void PipeBarrier() {}

struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, rsv; };
template <typename T> struct DataCopyPadExtParams { bool isPad; uint8_t left, right; T value; };

template <typename T>
void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, DataCopyExtParams p,
                 DataCopyPadExtParams<T> pad) {
    if (!p.blockCount || p.blockCount > 4095 || p.blockLen != sizeof(T) || p.srcStride % sizeof(T)
        || p.dstStride || pad.left || pad.right) throw std::runtime_error("DMA contract");
    if (src.IsInput()) ++counts->inputDma;
    for (size_t i = 0; i < p.blockCount; ++i) {
        const auto slot = i * 32 / sizeof(T);
        dst.SetValue(slot, src.GetValue(i * (p.blockLen + p.srcStride) / sizeof(T)));
        for (size_t j = 1; j < 32 / sizeof(T); ++j) {
            dst.SetValue(slot + j, static_cast<T>(std::numeric_limits<float>::quiet_NaN()));
        }
    }
}
template <typename T>
void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, DataCopyExtParams p) {
    if (!p.blockCount || p.blockCount > 4095 || p.blockLen != sizeof(T) || p.srcStride || p.dstStride % sizeof(T)) {
        throw std::runtime_error("DMA write contract");
    }
    for (size_t i = 0; i < p.blockCount; ++i) {
        dst.SetValue(i * (p.blockLen + p.dstStride) / sizeof(T), src.GetValue(i * 32 / sizeof(T)));
    }
}
template <typename T> void CreateVecIndex(LocalTensor<T> dst, T first, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) dst.SetValue(i, first + static_cast<T>(i));
}
template <typename T> void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, T shift, int32_t count) {
    for (int32_t i = 0; i < count; ++i) dst.SetValue(i, src.GetValue(i) << shift);
}
template <typename T>
void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& offset,
            uint32_t base, uint32_t count) {
    for (size_t i = 0; i < count; ++i) {
        const auto bytes = base + offset.GetValue(i);
        if (bytes % sizeof(T)) throw std::runtime_error("unaligned Gather");
        dst.SetValue(i, src.GetValue(bytes / sizeof(T)));
    }
}
template <typename T, typename U>
void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, RoundMode, uint32_t count) {
    for (size_t i = 0; i < count; ++i) dst.SetValue(i, static_cast<T>(src.GetValue(i)));
}
template <typename T>
void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& scratch,
               uint32_t count) {
    for (size_t i = 0; i < count; ++i) scratch.SetValue(i, src.GetValue(i));
    for (size_t n = count; n > 1; n = (n + 1) / 2) {
        for (size_t i = 0; i < n / 2; ++i) {
            scratch.SetValue(i, scratch.GetValue(2 * i) + scratch.GetValue(2 * i + 1));
        }
        if (n % 2) scratch.SetValue(n / 2, scratch.GetValue(n - 1));
    }
    dst.SetValue(0, scratch.GetValue(0));
}
template <typename T> void Duplicate(const LocalTensor<T>& dst, T value, uint32_t count) {
    for (size_t i = 0; i < count; ++i) dst.SetValue(i, value);
}
} // namespace AscendC
