#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace ge {
enum graphStatus { GRAPH_SUCCESS, GRAPH_FAILED };
enum DataType { DT_FLOAT16, DT_FLOAT, DT_BF16 };
} // namespace ge

namespace fe { struct PlatFormInfos { uint32_t aivCores = 24; }; }

namespace gert {
class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}
    size_t GetDimNum() const { return dims_.size(); }
    int64_t GetDim(size_t index) const { return dims_.at(index); }
    void SetDimNum(size_t n) { dims_.resize(n); }
    void SetDim(size_t i, int64_t dim) { dims_.at(i) = dim; }
private:
    std::vector<int64_t> dims_;
};

struct InferShapeContext {
    const Shape* input;
    Shape* output;
    const Shape* GetInputShape(size_t) const { return input; }
    Shape* GetOutputShape(size_t) const { return output; }
};
struct StorageShape {
    Shape storage;
    const Shape& GetStorageShape() const { return storage; }
};
struct TensorDesc {
    ge::DataType dtype;
    ge::DataType GetDataType() const { return dtype; }
};
struct TilingContext {
    const StorageShape* input;
    const TensorDesc* desc;
    void* tiling;
    size_t workspace = 1;
    uint32_t blockDim = 0;
    uint32_t scheduleMode = 0;
    uint64_t key = 999;
    fe::PlatFormInfos platform;
    const StorageShape* GetInputShape(size_t) const { return input; }
    const TensorDesc* GetInputDesc(size_t) const { return desc; }
    template <typename T> T* GetTilingData() const { return static_cast<T*>(tiling); }
    size_t* GetWorkspaceSizes(size_t) { return &workspace; }
    void SetBlockDim(uint32_t n) { blockDim = n; }
    ge::graphStatus SetScheduleMode(uint32_t mode) { scheduleMode = mode; return ge::GRAPH_SUCCESS; }
    void SetTilingKey(uint64_t n) { key = n; }
    fe::PlatFormInfos* GetPlatformInfo() { return &platform; }
};
struct TilingParseContext {};
} // namespace gert

struct MockRegistration {
    template <typename F> MockRegistration InferShape(F) const { return {}; }
    template <typename F> MockRegistration Tiling(F) const { return {}; }
    template <typename, typename F> MockRegistration TilingParse(F) const { return {}; }
};
#define IMPL_OP_INFERSHAPE(name) [[maybe_unused]] static auto mockInfer = MockRegistration{}
#define IMPL_OP_OPTILING(name) [[maybe_unused]] static auto mockTiling = MockRegistration{}
