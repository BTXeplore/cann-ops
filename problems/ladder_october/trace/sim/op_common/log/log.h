#pragma once
#define OP_LOGE(...)
#define OP_CHECK_NULL_WITH_CONTEXT(context, value) do { if ((value) == nullptr) return ge::GRAPH_FAILED; } while (false)
#define OP_CHECK_IF(condition, log, action) do { if (condition) { action; } } while (false)
