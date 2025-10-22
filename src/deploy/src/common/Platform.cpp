#include "common/Platform.hpp"
#include "utils/Log.hpp"
#include <cassert>

// 瑞芯微RKNN SDK头文件（实际项目中需配置SDK路径）
#ifdef RKNN_SUPPORT
#include "rknn_api.h"
#endif

namespace deploy {
namespace common {

// 计算单元→字符串映射表
static const std::unordered_map<ComputeUnit, std::string> kUnitToString = {
    {ComputeUnit::UNKNOWN, "unknown"},
    {ComputeUnit::CPU, "cpu"},
    {ComputeUnit::RK_NPU, "rk_npu"},
    {ComputeUnit::RK_VPU, "rk_vpu"},
    {ComputeUnit::NVIDIA_CUDA, "nvidia_cuda"},
    {ComputeUnit::NVIDIA_DLA, "nvidia_dla"},
    {ComputeUnit::ASCEND_NPU, "ascend_npu"},
};

// 字符串→计算单元映射表
static const std::unordered_map<std::string, ComputeUnit> kStringToUnit = {
    {"unknown", ComputeUnit::UNKNOWN},
    {"cpu", ComputeUnit::CPU},
    {"rk_npu", ComputeUnit::RK_NPU},
    {"rk_vpu", ComputeUnit::RK_VPU},
    {"nvidia_cuda", ComputeUnit::NVIDIA_CUDA},
    {"nvidia_dla", ComputeUnit::NVIDIA_DLA},
    {"ascend_npu", ComputeUnit::ASCEND_NPU},
};

// 计算单元→芯片平台映射表
static const std::unordered_map<ComputeUnit, ChipPlatform> kUnitToChip = {
    {ComputeUnit::CPU, ChipPlatform::GENERIC},
    {ComputeUnit::RK_NPU, ChipPlatform::ROCKCHIP},
    {ComputeUnit::RK_VPU, ChipPlatform::ROCKCHIP},
    {ComputeUnit::NVIDIA_CUDA, ChipPlatform::NVIDIA},
    {ComputeUnit::NVIDIA_DLA, ChipPlatform::NVIDIA},
    {ComputeUnit::ASCEND_NPU, ChipPlatform::ASCEND},
};

// DeviceId转字符串
std::string DeviceId::to_string() const {
    return compute_unit_to_string(unit) + ":" + std::to_string(id);
}

// 计算单元转字符串
std::string compute_unit_to_string(ComputeUnit unit) {
    auto it = kUnitToString.find(unit);
    if (it != kUnitToString.end()) {
        return it->second;
    }
    LOG_WARN("未知计算单元类型: {}", static_cast<int>(unit));
    return "unknown";
}

// 字符串转计算单元
ComputeUnit string_to_compute_unit(const std::string& str) {
    auto it = kStringToUnit.find(str);
    if (it != kStringToUnit.end()) {
        return it->second;
    }
    LOG_WARN("未知计算单元字符串: {}", str);
    return ComputeUnit::UNKNOWN;
}

// 获取计算单元所属芯片平台
ChipPlatform get_chip_platform(ComputeUnit unit) {
    auto it = kUnitToChip.find(unit);
    if (it != kUnitToChip.end()) {
        return it->second;
    }
    return ChipPlatform::UNKNOWN;
}

// 获取设备属性（核心实现：调用各平台SDK查询）
DeviceProperties get_device_properties(const DeviceId& device_id) {
    DeviceProperties prop;
    prop.unit = device_id.unit;
    prop.chip_platform = get_chip_platform(device_id.unit);
    prop.chip_model = "unknown";
    prop.unit_model = "unknown";
    prop.memory_size = 0;
    prop.max_batch_size = 1;
    prop.supports_preprocess = true;
    prop.supports_postprocess = true;
    prop.supports_infer = true;

    // 针对不同计算单元，调用对应的SDK获取属性
    switch (device_id.unit) {
        case ComputeUnit::RK_NPU:
#ifdef RKNN_SUPPORT
            // 调用RKNN SDK查询设备信息（实际项目需初始化RKNN上下文）
            rknn_dev_info dev_info;
            memset(&dev_info, 0, sizeof(dev_info));
            int ret = rknn_query(nullptr, RKNN_QUERY_DEV_INFO, &dev_info, sizeof(dev_info));
            if (ret == 0) {
                prop.chip_model = dev_info.model;  // 如"RK3588"
                prop.unit_model = "RK NPU";
                prop.memory_size = dev_info.memory_size;  // NPU专用内存
                prop.max_batch_size = 4;  // RK3588支持批次4，RK3568可能为2
                // 根据芯片型号细化属性
                if (prop.chip_model == "RK3568") {
                    prop.max_batch_size = 2;
                    prop.supported_ops = {"conv2d", "relu", "pooling"};  // 简化示例
                } else if (prop.chip_model == "RK3588") {
                    prop.max_batch_size = 4;
                    prop.supported_ops = {"conv2d", "relu", "pooling", "sigmoid", "transpose"};
                }
            } else {
                LOG_ERROR("RKNN设备信息查询失败");
            }
#else
            LOG_WARN("未启用RKNN支持，使用默认属性");
            prop.chip_model = "RK3588";  // 默认假设
            prop.memory_size = 2 * 1024 * 1024 * 1024;  // 2GB
#endif
            break;

        case ComputeUnit::CPU:
            prop.chip_model = "Generic CPU";
            prop.unit_model = "x86/ARM CPU";
            prop.memory_size = 8 * 1024 * 1024 * 1024;  // 假设8GB内存
            prop.max_batch_size = 8;
            prop.supported_ops = {"all"};  // CPU支持所有算子（软件实现）
            break;

        case ComputeUnit::NVIDIA_CUDA:
            // 实际项目中调用CUDA SDK查询（cudaGetDeviceProperties）
            prop.chip_model = "NVIDIA Orin";
            prop.unit_model = "CUDA Core";
            prop.memory_size = 16 * 1024 * 1024 * 1024;  // 16GB
            prop.max_batch_size = 16;
            prop.supported_ops = {"conv2d", "relu", "softmax", "nms"};
            break;

        default:
            LOG_WARN("未实现计算单元{}的属性查询", compute_unit_to_string(device_id.unit));
            break;
    }

    return prop;
}

}  // namespace common
}  // namespace deploy