#pragma once
#include "host_api.h"
namespace platform_ascendc {
enum class CoreMemType { UB };
struct PlatformAscendC {
    explicit PlatformAscendC(fe::PlatFormInfos* info) : info_(info) {}
    uint32_t GetCoreNumAiv() const { return info_->aivCores; }
    void GetCoreMemSize(CoreMemType, uint64_t& bytes) const { bytes = 196608; }
private:
    fe::PlatFormInfos* info_;
};
} // namespace platform_ascendc
