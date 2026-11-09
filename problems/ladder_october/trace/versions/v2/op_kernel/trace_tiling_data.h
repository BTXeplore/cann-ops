#ifndef _TRACE_TILING_DATA_H_
#define _TRACE_TILING_DATA_H_

#include <stdint.h>

constexpr uint32_t kTraceTile = 512;
constexpr uint32_t kTraceMaxDim = 16384;

struct TraceTilingData {
    int64_t totalNum = 0;     // Number of diagonal elements.
    int64_t blockFactor = 1;  // Distance between diagonal elements in GM.
    int64_t ubFactor = 0;     // Diagonal elements processed per tile.
};

inline bool MakeTraceTiling(int64_t rows, int64_t columns, TraceTilingData& tiling)
{
    if (rows < 1 || columns < 1 || rows > kTraceMaxDim || columns > kTraceMaxDim) return false;
    tiling.totalNum = rows < columns ? rows : columns;
    tiling.blockFactor = columns + 1;
    tiling.ubFactor = tiling.totalNum < kTraceTile ? tiling.totalNum : kTraceTile;
    return true;
}

#endif
