#ifndef TRACE_H
#define TRACE_H

#include <stdint.h>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "trace_tiling_data.h"
#include "trace_tiling_key.h"

namespace NsTrace {
using namespace AscendC;

// The local test builds a scalar variant by changing only this constant.
constexpr bool kStridedDma = true;
constexpr uint32_t kTile = 128;

template <typename T>
class Trace {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, const TraceTilingData* tiling)
    {
        count_ = static_cast<uint32_t>(tiling->totalNum);
        stride_ = static_cast<uint32_t>(tiling->blockFactor);
        tile_ = static_cast<uint32_t>(tiling->ubFactor);
        const uint32_t span = (count_ - 1) * stride_ + 1;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), span);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), 1);
        pipe_.InitBuffer(outputQueue_, 1, 32);
        if constexpr (kStridedDma) {
            pipe_.InitBuffer(inputQueue_, 1, kTile * 32);
            pipe_.InitBuffer(offsets_, kTile * sizeof(uint32_t));
            pipe_.InitBuffer(packed_, kTile * sizeof(T));
            pipe_.InitBuffer(fp32_, kTile * sizeof(float));
            pipe_.InitBuffer(work_, kTile * sizeof(float));
            pipe_.InitBuffer(partial_, 32);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (kStridedDma) {
            if (count_ == 1) {
                // A one-element DMA preserves the output bit pattern, including -0.
                auto local = inputQueue_.AllocTensor<T>();
                DataCopyPad(local, input_, DataCopyExtParams{1, sizeof(T), 0, 0, 0},
                            DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0.0f)});
                inputQueue_.EnQue(local);
                local = inputQueue_.DeQue<T>();
                const auto ready = pipe_.FetchEventID(HardEvent::MTE2_S);
                SetFlag<HardEvent::MTE2_S>(ready);
                WaitFlag<HardEvent::MTE2_S>(ready);
                Write(local.GetValue(0));
                inputQueue_.FreeTensor(local);
                return;
            }
            Write(static_cast<T>(SumTiles()));
        } else {
            // Single-core correctness reference. Read only diagonal locations.
            float sum = static_cast<float>(input_.GetValue(0));
            for (uint32_t i = 1; i < count_; ++i) {
                sum += static_cast<float>(input_.GetValue(static_cast<uint64_t>(i) * stride_));
            }
            Write(static_cast<T>(sum));
        }
    }

private:
    __aicore__ inline float SumTiles()
    {
        auto offsets = offsets_.Get<uint32_t>();
        for (uint32_t i = 0; i < tile_; ++i) offsets.SetValue(i, i * 32);
        const auto offsetsReady = pipe_.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(offsetsReady);
        WaitFlag<HardEvent::S_V>(offsetsReady);

        float total = 0.0f;
        for (uint32_t start = 0; start < count_; start += tile_) {
            const uint32_t size = count_ - start < tile_ ? count_ - start : tile_;
            auto sparse = inputQueue_.AllocTensor<T>();
            // GM gap is in bytes after the copied element. UB slots round up to 32 B.
            DataCopyExtParams copy{static_cast<uint16_t>(size), sizeof(T),
                                   static_cast<uint32_t>((stride_ - 1) * sizeof(T)), 0, 0};
            DataCopyPad(sparse, input_[static_cast<uint64_t>(start) * stride_], copy,
                        DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0.0f)});
            inputQueue_.EnQue(sparse);
            sparse = inputQueue_.DeQue<T>();
            auto compact = packed_.Get<T>();
            Gather(compact, sparse, offsets, 0U, size);
            PipeBarrier<PIPE_V>();
            inputQueue_.FreeTensor(sparse);
            auto partial = partial_.Get<float>();
            auto scratch = work_.Get<float>();
            if constexpr (sizeof(T) == sizeof(half)) {
                auto f32 = fp32_.Get<float>();
                Cast(f32, compact, RoundMode::CAST_NONE, size);
                PipeBarrier<PIPE_V>();
                ReduceSum(partial, f32, scratch, size);
            } else {
                ReduceSum(partial, compact, scratch, size);
            }
            const auto sumReady = pipe_.FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(sumReady);
            WaitFlag<HardEvent::V_S>(sumReady);
            total += partial.GetValue(0);
        }
        return total;
    }

    __aicore__ inline void Write(T value)
    {
        auto local = outputQueue_.AllocTensor<T>();
        const auto valueReady = pipe_.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(valueReady);
        WaitFlag<HardEvent::S_V>(valueReady);
        Duplicate(local, value, 32 / sizeof(T));
        outputQueue_.EnQue(local);
        local = outputQueue_.DeQue<T>();
        DataCopyPad(output_, local, DataCopyExtParams{1, sizeof(T), 0, 0, 0});
        outputQueue_.FreeTensor(local);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> offsets_, packed_, fp32_, work_, partial_;
    GlobalTensor<T> input_, output_;
    uint32_t count_ = 0, stride_ = 0, tile_ = 0;
};
} // namespace NsTrace
#endif
