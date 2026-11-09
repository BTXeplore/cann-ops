#include "register/op_def_registry.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/trace_tiling_data.h"
#include "../op_kernel/trace_tiling_key.h"

namespace optiling {
constexpr uint32_t kTilesPerCore = 2;
constexpr uint32_t kMinParallelTiles = 4;
constexpr uint32_t kPartialSlotBytes = 32;

static ge::graphStatus TraceTilingFunc(gert::TilingContext* context)
{
    const auto* input = context->GetInputShape(0);
    const auto* desc = context->GetInputDesc(0);
    auto* tiling = context->GetTilingData<TraceTilingData>();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (input == nullptr || desc == nullptr || tiling == nullptr || workspace == nullptr) return ge::GRAPH_FAILED;
    const auto& shape = input->GetStorageShape();
    if (shape.GetDimNum() != 2 || !MakeTraceTiling(shape.GetDim(0), shape.GetDim(1), *tiling)) return ge::GRAPH_FAILED;
    const auto dtype = desc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    const uint32_t tiles = static_cast<uint32_t>((tiling->totalNum + tiling->ubFactor - 1) / tiling->ubFactor);
    uint32_t cores = 1;
    if (tiles >= kMinParallelTiles) {
        auto* platformInfo = context->GetPlatformInfo();
        if (platformInfo != nullptr) {
            auto platform = platform_ascendc::PlatformAscendC(platformInfo);
            const uint32_t available = platform.GetCoreNumAiv();
            if (available > 0) {
                const uint32_t requested = tiles / kTilesPerCore;
                cores = requested < available ? requested : available;
            }
        }
    }
    // Batch scheduling reserves all participating AIV cores before SyncAll.
    if (cores > 1 && context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    workspace[0] = cores > 1 ? tiles * kPartialSlotBytes : 0;
    context->SetBlockDim(cores);
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ? GET_TPL_TILING_KEY(TRACE_TPL_SCH_MODE_0)
                                                  : GET_TPL_TILING_KEY(TRACE_TPL_SCH_MODE_1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTrace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TraceCompileInfo {};
IMPL_OP_OPTILING(Trace).Tiling(TraceTilingFunc).TilingParse<TraceCompileInfo>(TilingParseForTrace);
} // namespace optiling
