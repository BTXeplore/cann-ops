#ifndef _TRACE_TILING_DATA_H_
#define _TRACE_TILING_DATA_H_

#include <stdint.h>

struct TraceTilingData {
    int64_t totalNum = 0;     // Number of diagonal elements.
    int64_t blockFactor = 1;  // Distance between diagonal elements in GM.
    int64_t ubFactor = 0;     // Diagonal elements processed per tile.
};

inline bool MakeTraceTiling(int64_t rows, int64_t columns, TraceTilingData& tiling)
{
    if (rows < 1 || columns < 1 || rows > 16384 || columns > 16384) return false;
    tiling.totalNum = rows < columns ? rows : columns;
    tiling.blockFactor = columns + 1;
    tiling.ubFactor = tiling.totalNum < 128 ? tiling.totalNum : 128;
    return true;
}

#endif
