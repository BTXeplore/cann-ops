#include "register/op_def_registry.h"
#include "../op_kernel/trace_tiling_data.h"
#include "../op_kernel/trace_tiling_key.h"

namespace optiling {
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
    workspace[0] = 0;
    context->SetBlockDim(1);
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
