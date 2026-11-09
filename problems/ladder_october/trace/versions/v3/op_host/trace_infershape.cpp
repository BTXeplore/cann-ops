#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
static ge::graphStatus InferShapeTrace(gert::InferShapeContext* context)
{
    const auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr || input->GetDimNum() != 2) return ge::GRAPH_FAILED;
    output->SetDimNum(1);
    output->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Trace).InferShape(InferShapeTrace);
} // namespace ops
