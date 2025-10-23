#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#include <string>
#include <cstdint>  // 用于uint64_t

namespace deploy {
namespace common {

// 计算单元类型（支持的硬件加速单元）
enum class ComputeUnit {
    CPU,         // 通用CPU
    RK_NPU,      // 瑞芯微NPU
    RK_VPU,      // 瑞芯微VPU
    NVIDIA_CUDA, // NVIDIA GPU (CUDA)
    NVIDIA_DLA,  // NVIDIA DLA
    ASCEND_NPU,  // 华为昇腾NPU
    UNKNOWN      // 未知类型
};

// 芯片平台（厂商）
enum class ChipPlatform {
    GENERIC,    // 通用（如CPU）
    ROCKCHIP,   // 瑞芯微
    NVIDIA,     // NVIDIA
    ASCEND,     // 华为昇腾
    UNKNOWN     // 未知
};

// 设备ID（计算单元类型 + 设备编号）
struct DeviceId {
    ComputeUnit unit;  // 计算单元类型
    int id;            // 设备编号（如CPU:0, RK_NPU:1）

    DeviceId(ComputeUnit u, int i) : unit(u), id(i) {}

    // 转换为字符串（如"rk_npu:0"）
    std::string to_string() const;
};

// 设备属性（算力、内存、支持的功能等）
struct DeviceProperties {
    ComputeUnit unit;              // 计算单元类型
    ChipPlatform chip_platform;    // 芯片平台
    std::string chip_model;        // 芯片型号（如"RK3588", "NVIDIA A100"）
    uint64_t memory_size;          // 内存大小（字节）
    int max_batch_size;            // 最大批次大小
    bool supports_preprocess;      // 是否支持预处理
    bool supports_infer;           // 是否支持推理
    bool supports_postprocess;     // 是否支持后处理
};

// 计算单元 ↔ 字符串 转换
std::string compute_unit_to_string(ComputeUnit unit);
ComputeUnit string_to_compute_unit(const std::string& str);

// 根据计算单元获取芯片平台
ChipPlatform get_chip_platform(ComputeUnit unit);

// 查询设备属性
DeviceProperties get_device_properties(const DeviceId& device_id);

}  // namespace common
}  // namespace deploy

#endif  // PLATFORM_HPP