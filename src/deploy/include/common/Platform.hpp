#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace deploy {
namespace common {

// 1. 硬件计算单元类型（芯片内的具体加速模块）
enum class ComputeUnit : int32_t {
    UNKNOWN = 0,

    // 通用计算单元
    CPU = 1,  // 中央处理器（x86/ARM）

    // 瑞芯微（Rockchip）计算单元
    RK_NPU = 2,  // RK系列NPU（如RK3588/RK3568的NPU）
    RK_VPU = 3,  // RK系列VPU（视频处理单元）

    // NVIDIA计算单元
    NVIDIA_CUDA = 4,  // NVIDIA GPU的CUDA核心
    NVIDIA_DLA = 5,   // NVIDIA深度学习加速器（如Orin DLA）

    // 华为昇腾计算单元
    ASCEND_NPU = 6    // 昇腾NPU（如Ascend 310B）
};

// 2. 计算单元所属的芯片平台
enum class ChipPlatform {
    UNKNOWN,
    ROCKCHIP,    // 瑞芯微
    NVIDIA,      // NVIDIA
    ASCEND,      // 华为昇腾
    GENERIC      // 通用（如CPU）
};

// 3. 设备ID（唯一标识计算单元实例）
struct DeviceId {
    ComputeUnit unit = ComputeUnit::UNKNOWN;  // 计算单元类型
    int32_t id = 0;                           // 设备编号（多设备时区分）

    bool operator==(const DeviceId& other) const {
        return (unit == other.unit) && (id == other.id);
    }

    // 转字符串（如"rk_npu:0"）
    std::string to_string() const;
};

// 4. 设备详细属性（运行时动态获取）
struct DeviceProperties {
    ComputeUnit unit;                  // 计算单元类型
    ChipPlatform chip_platform;        // 所属芯片平台
    std::string chip_model;            // 芯片型号（如"RK3588"、"Orin NX"）
    std::string unit_model;            // 计算单元型号（如"RK3588 NPU v2"）
    size_t memory_size;                // 专用内存大小（字节）
    int max_batch_size;                // 最大批次大小
    std::vector<std::string> supported_ops;  // 支持的算子列表
    bool supports_preprocess;          // 是否支持预处理加速
    bool supports_postprocess;         // 是否支持后处理加速
    bool supports_infer;               // 是否支持推理加速
};

// 5. 辅助函数声明
// 计算单元 ↔ 字符串转换
std::string compute_unit_to_string(ComputeUnit unit);
ComputeUnit string_to_compute_unit(const std::string& str);

// 获取计算单元所属的芯片平台
ChipPlatform get_chip_platform(ComputeUnit unit);

// 获取设备属性（通过底层SDK查询）
DeviceProperties get_device_properties(const DeviceId& device_id);

}  // namespace common
}  // namespace deploy

#endif  // PLATFORM_HPP