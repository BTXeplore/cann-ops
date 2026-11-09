#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include "trace.cpp"
#include "trace_infershape.cpp"
#include "trace_tiling.cpp"

template <typename T> GM_ADDR address(T& value) { return reinterpret_cast<GM_ADDR>(&value); }

struct KernelCounts {
    size_t reads = 0, inputDma = 0, maxInputDma = 0, vectorToScalar = 0;
    bool offDiagonal = false;
};

template <uint32_t Mode>
KernelCounts launch(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling,
                    uint32_t blocks, size_t rotation) {
    std::vector<AscendC::WorkCounts> counts(blocks);
    std::vector<std::exception_ptr> errors(blocks);
    std::barrier barrier(static_cast<std::ptrdiff_t>(blocks));
    AscendC::blockCount = blocks;
    AscendC::blockBarrier = blocks > 1 ? &barrier : nullptr;
    const auto worker = [&](uint32_t index) {
        AscendC::blockIndex = index;
        AscendC::counts = &counts[index];
        try {
            trace<Mode>(input, output, workspace, tiling);
        } catch (...) {
            errors[index] = std::current_exception();
            if (blocks > 1) barrier.arrive_and_drop();
        }
        AscendC::counts = nullptr;
    };
    if (blocks == 1) {
        worker(0);
    } else {
        std::vector<std::jthread> workers;
        workers.reserve(blocks);
        for (uint32_t i = 0; i < blocks; ++i) {
            workers.emplace_back(worker, static_cast<uint32_t>((i + rotation) % blocks));
        }
    }
    AscendC::blockBarrier = nullptr;
    AscendC::blockCount = 1;
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    KernelCounts total;
    for (const auto& block : counts) {
        total.reads += block.reads;
        total.inputDma += block.inputDma;
        total.maxInputDma = std::max(total.maxInputDma, block.inputDma);
        total.vectorToScalar += block.vectorToScalar;
        total.offDiagonal |= block.readOffDiagonal;
    }
    return total;
}

template <typename T>
void run(const char* id, int rows, int columns, const std::string& pattern = "random", uint32_t cores = 24) {
    T marker{};
    auto& matrix = AscendC::matrix<T>;
    matrix = {&marker, static_cast<size_t>(columns) + 1, static_cast<size_t>(rows) * columns};
    std::mt19937 rng(102938 + rows * 17 + columns);
    for (int i = 0; i < std::min(rows, columns); ++i) {
        const int value = static_cast<int>(rng() % 2049) - 1024;
        matrix.diagonal.push_back(static_cast<T>(value / 1024.0f));
    }
    if (pattern == "random-frac") {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : matrix.diagonal) v = static_cast<T>(dist(rng));
    }
    if (pattern == "negative-zero") matrix.diagonal[0] = static_cast<T>(-0.0f);
    if (pattern == "example-5") matrix.diagonal = {static_cast<T>(1), static_cast<T>(4)};
    if (pattern == "example-6") matrix.diagonal = {static_cast<T>(1), static_cast<T>(5)};
    if (pattern == "nan") matrix.diagonal.back() = static_cast<T>(NAN);
    if (pattern == "positive-inf" || pattern == "opposing-inf") matrix.diagonal[0] = static_cast<T>(INFINITY);
    if (pattern == "negative-inf" || pattern == "opposing-inf") matrix.diagonal.back() = static_cast<T>(-INFINITY);
    if (pattern == "cancel") {
        for (size_t i = 0; i < matrix.diagonal.size(); ++i) matrix.diagonal[i] = static_cast<T>(i % 2 ? -1.0 : 1.0);
        matrix.diagonal.back() = static_cast<T>(-0.9990234375);
    }
    matrix.visits.resize(matrix.diagonal.size());
    double reference = static_cast<double>(matrix.diagonal[0]);
    for (size_t i = 1; i < matrix.diagonal.size(); ++i) reference += static_cast<double>(matrix.diagonal[i]);
    const double expected = static_cast<double>(static_cast<T>(reference));
    constexpr bool isHalf = std::is_same_v<T, half>;
    const gert::StorageShape inputShape{{rows, columns}};
    gert::Shape outputShape;
    const gert::TensorDesc desc{isHalf ? ge::DT_FLOAT16 : ge::DT_FLOAT};
    gert::InferShapeContext shapeContext{&inputShape.storage, &outputShape};
    TraceTilingData tiling;
    gert::TilingContext tilingContext{&inputShape, &desc, &tiling};
    tilingContext.platform.aivCores = cores;
    const bool shapeOk = ops::InferShapeTrace(&shapeContext) == ge::GRAPH_SUCCESS
                         && outputShape.GetDimNum() == 1 && outputShape.GetDim(0) == 1;
    const bool validDims = optiling::TraceTilingFunc(&tilingContext) == ge::GRAPH_SUCCESS
                           && tiling.totalNum == static_cast<int64_t>(matrix.diagonal.size())
                           && tiling.blockFactor == columns + 1
                           && tiling.ubFactor > 0 && tiling.ubFactor <= tiling.totalNum && tiling.ubFactor <= 512;
    const auto blocks = tilingContext.blockDim;
    const auto tiles = validDims ? (tiling.totalNum + tiling.ubFactor - 1) / tiling.ubFactor : 0;
    const bool workspaceOk = blocks == 1 ? tilingContext.workspace == 0 && tilingContext.scheduleMode == 0
                                         : tilingContext.workspace == static_cast<size_t>(tiles) * 32
                                               && tilingContext.scheduleMode == 1;
    const bool tilingOk = validDims && blocks > 0 && blocks <= std::max(cores, 1U)
                          && blocks <= tiles && tilingContext.key == (isHalf ? 0U : 1U) && workspaceOk;
    if (!tilingOk) throw std::runtime_error(std::string("Invalid host tiling for ") + id);
    constexpr double limit = isHalf ? 1e-3 : 1e-4;
    bool correct = true;
    bool identical = true;
    T previous{};
    double actual = 0;
    uint32_t outputBits = 0;
    KernelCounts counts;
    std::array<long long, 5> samples{};
    for (size_t attempt = 0; attempt < samples.size(); ++attempt) {
        std::array<T, 3> output{static_cast<T>(-8), static_cast<T>(-937), static_cast<T>(22)};
        constexpr float guard = -1e20f;
        constexpr size_t guardSize = 8;
        const size_t workspaceSize = tilingContext.workspace / sizeof(float);
        std::vector<float> workspace(workspaceSize + 2 * guardSize, guard);
        std::fill(matrix.visits.begin(), matrix.visits.end(), 0);
        auto begin = std::chrono::steady_clock::now();
        counts = launch<isHalf ? 0 : 1>(address(marker), address(output[1]),
                                       workspaceSize ? address(workspace[guardSize]) : nullptr,
                                       address(tiling), blocks, attempt);
        samples[attempt] = std::max(1LL, static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count()));
        actual = static_cast<double>(output[1]);
        std::memcpy(&outputBits, &output[1], sizeof(T));
        if (attempt && std::memcmp(&previous, &output[1], sizeof(T)) != 0) identical = false;
        previous = output[1];
        bool match;
        if (std::isnan(expected)) match = std::isnan(actual);
        else if (std::isinf(expected)) match = expected == actual;
        else {
            const double error = std::abs(actual - expected);
            match = error < limit && (std::abs(expected) < limit || error / std::abs(expected) < limit);
            if (pattern == "negative-zero") match = match && std::signbit(actual);
        }
        bool canaries = output[0] == static_cast<T>(-8) && output[2] == static_cast<T>(22);
        for (size_t i = 0; i < workspace.size(); ++i) {
            const bool isPartial = i >= guardSize && i < guardSize + workspaceSize && (i - guardSize) % 8 == 0;
            canaries = canaries && (isPartial ? workspace[i] != guard : workspace[i] == guard);
        }
        correct = correct && match && !counts.offDiagonal && counts.reads == matrix.diagonal.size() && canaries
                  && std::all_of(matrix.visits.begin(), matrix.visits.end(), [](uint32_t n) { return n == 1; });
    }
    std::cout << id << '\t' << rows << '\t' << columns << '\t' << (isHalf ? "float16" : "float32")
              << '\t' << (correct && identical && shapeOk && tilingOk) << '\t' << identical << '\t' << expected << '\t' << actual
              << '\t' << counts.reads << '\t' << counts.offDiagonal << '\t' << correct
              << '\t' << shapeOk << '\t' << tilingOk << '\t' << outputBits << '\t' << blocks
              << '\t' << tiling.ubFactor << '\t' << tilingContext.workspace << '\t' << counts.inputDma
              << '\t' << counts.maxInputDma << '\t' << counts.vectorToScalar;
    for (auto ns : samples) std::cout << '\t' << ns;
    std::cout << '\n';
}

int main() {
    std::cout << std::setprecision(17);
    run<float>("example-f32", 2, 2, "example-5");
    run<half>("example-f16", 2, 3, "example-6");
    run<float>("unit-f32", 1, 1);
    run<half>("negative-zero-f16", 1, 1, "negative-zero");
    run<float>("row-max-f32", 1, 16384);
    run<half>("col-max-f16", 16384, 1);
    run<float>("wide-f32", 129, 16384);
    run<half>("tall-f16", 8193, 63);
    run<float>("tail-127-f32", 127, 131);
    run<float>("tile-128-f32", 128, 130);
    run<float>("tail-129-f32", 129, 131);
    run<half>("tail-255-f16", 256, 255);
    run<half>("tile-256-f16", 256, 256);
    run<half>("tail-257-f16", 257, 261);
    run<half>("limit-2048-f16", 2048, 2051);
    run<float>("max-square-f32", 16384, 16384);
    run<float>("fractional-8193-f32", 8193, 8195, "random-frac");
    run<half>("fractional-2048-f16", 2053, 2048, "random-frac");
    run<float>("near-zero-f32", 1024, 1025, "cancel");
    run<half>("near-zero-f16", 256, 257, "cancel");
    run<float>("diag-nan-f32", 193, 197, "nan");
    run<half>("diag-nan-f16", 19, 23, "nan");
    run<float>("positive-inf-f32", 130, 134, "positive-inf");
    run<half>("negative-inf-f16", 13, 17, "negative-inf");
    run<float>("opposing-inf-f32", 129, 131, "opposing-inf");
    run<half>("opposing-inf-f16", 129, 131, "opposing-inf");
    run<float>("tail-511-f32", 511, 521, "random-frac");
    run<float>("tile-512-f32", 512, 513, "random-frac");
    run<float>("tail-513-f32", 513, 519, "random-frac");
    run<float>("last-single-core-f32", 1536, 1553, "random-frac");
    run<float>("first-multi-core-f32", 1537, 1553, "random-frac");
    run<float>("cross-core-inf-f32", 2051, 2063, "opposing-inf");
    run<half>("cross-core-nan-f16", 2039, 2041, "nan");
    run<float>("core-limit-three-f32", 12289, 16384, "random-frac", 3);
    run<float>("core-limit-one-f32", 8193, 16384, "random-frac", 1);
    run<float>("no-platform-cores-f32", 4097, 4100, "random-frac", 0);
}
