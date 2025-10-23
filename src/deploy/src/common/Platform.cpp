#include "common/Platform.hpp"
#include <unordered_map>

namespace deploy {
namespace common {

// 计算单元到字符串的映射
static const std::unordered_map<ComputeUnit, std::string> kUnitToString = {
    {ComputeUnit::CPU, "cpu"},
    {ComputeUnit::RK_NPU, "rk_npu"},
    {ComputeUnit::RK_VPU, "rk_vpu"},
    {ComputeUnit::NVIDIA_CUDA, "nvidia_cuda"},
    {ComputeUnit::NVIDIA_DLA, "nvidia_dla"},
    {ComputeUnit::ASCEND_NPU, "ascend_npu"},
    {ComputeUnit::UNKNOWN, "unknown"}
};

// 字符串到计算单元的映射
static const std::unordered_map<std::string, ComputeUnit> kStringToUnit = {
    {"cpu", ComputeUnit::CPU},
    {"rk_npu", ComputeUnit::RK_NPU},
    {"rk_vpu", ComputeUnit::RK_VPU},
    {"nvidia_cuda", ComputeUnit::NVIDIA_CUDA},
    {"nvidia_dla", ComputeUnit::NVIDIA_DLA},
    {"ascend_npu", ComputeUnit::ASCEND_NPU}
};

// 计算单元转换为字符串
std::string compute_unit_to_string(ComputeUnit unit) {
    auto it = kUnitToString.find(unit);
    if (it != kUnitToString.end()) {
        return it->second;
    }
    // 未知类型返回默认值
    return "unknown";
}

// 字符串转换为计算单元
ComputeUnit string_to_compute_unit(const std::string& str) {
    auto it = kStringToUnit.find(str);
    if (it != kStringToUnit.end()) {
        return it->second;
    }
    // 未知字符串返回UNKNOWN
    return ComputeUnit::UNKNOWN;
}

// DeviceId转换为字符串
std::string DeviceId::to_string() const {
    return compute_unit_to_string(unit) + ":" + std::to_string(id);
}

// 计算单元对应的芯片平台
ChipPlatform get_chip_platform(ComputeUnit unit) {
    switch (unit) {
        case ComputeUnit::CPU:         return ChipPlatform::GENERIC;
        case ComputeUnit::RK_NPU:
        case ComputeUnit::RK_VPU:      return ChipPlatform::ROCKCHIP;
        case ComputeUnit::NVIDIA_CUDA:
        case ComputeUnit::NVIDIA_DLA:  return ChipPlatform::NVIDIA;
        case ComputeUnit::ASCEND_NPU:  return ChipPlatform::ASCEND;
        default:                       return ChipPlatform::UNKNOWN;
    }
}

// 查询设备属性（核心逻辑）
DeviceProperties get_device_properties(const DeviceId& device_id) {
    DeviceProperties prop;
    prop.unit = device_id.unit;
    prop.chip_platform = get_chip_platform(device_id.unit);

    // 根据计算单元类型初始化属性
    switch (device_id.unit) {
        case ComputeUnit::CPU: {
            prop.chip_model = "Generic CPU";
            prop.memory_size = 8LL * 1024 * 1024 * 1024;  // 假设8GB内存
            prop.max_batch_size = 16;
            prop.supports_preprocess = true;
            prop.supports_infer = true;
            prop.supports_postprocess = true;
            break;
        }
        case ComputeUnit::RK_NPU: {
#ifdef RK_PLATFORM
            // 实际环境中可通过RKNN SDK获取真实属性
            prop.chip_model = "RK3588";  // 示例型号
            prop.memory_size = 2LL * 1024 * 1024 * 1024;  // 2GB NPU内存
            prop.max_batch_size = 4;
#else
            prop.chip_model = "RK (simulated)";
            prop.memory_size = 2LL * 1024 * 1024 * 1024;
            prop.max_batch_size = 2;
#endif
            prop.supports_preprocess = false;
            prop.supports_infer = true;
            prop.supports_postprocess = false;
            break;
        }
        case ComputeUnit::NVIDIA_CUDA: {
#ifdef NVIDIA_PLATFORM
            prop.chip_model = "NVIDIA A100";  // 示例型号
            prop.memory_size = 16LL * 1024 * 1024 * 1024;  // 16GB
            prop.max_batch_size = 32;
#else
            prop.chip_model = "NVIDIA (simulated)";
            prop.memory_size = 8LL * 1024 * 1024 * 1024;
            prop.max_batch_size = 8;
#endif
            prop.supports_preprocess = false;
            prop.supports_infer = true;
            prop.supports_postprocess = false;
            break;
        }
        default: {
            prop.chip_model = "unknown";
            prop.memory_size = 0;
            prop.max_batch_size = 1;
            prop.supports_preprocess = false;
            prop.supports_infer = false;
            prop.supports_postprocess = false;
        }
    }

    return prop;
}

}  // namespace common
}  // namespace deploy