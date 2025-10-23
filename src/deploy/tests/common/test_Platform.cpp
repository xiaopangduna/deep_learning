#include <gtest/gtest.h>
#include "common/Platform.hpp"
#include <unordered_map>
#include <cstdint>

namespace deploy {
namespace common {
namespace test {

// 测试1：计算单元 ↔ 字符串 转换（核心功能）
TEST(PlatformTest, ComputeUnitStringConversion) {
    // 定义所有枚举值与预期字符串的映射（覆盖所有case）
    const std::unordered_map<ComputeUnit, std::string> kExpectedMap = {
        {ComputeUnit::CPU, "cpu"},
        {ComputeUnit::RK_NPU, "rk_npu"},
        {ComputeUnit::RK_VPU, "rk_vpu"},
        {ComputeUnit::NVIDIA_CUDA, "nvidia_cuda"},
        {ComputeUnit::NVIDIA_DLA, "nvidia_dla"},
        {ComputeUnit::ASCEND_NPU, "ascend_npu"},
        {ComputeUnit::UNKNOWN, "unknown"}
    };

    // 测试枚举 → 字符串
    for (const auto& [unit, str] : kExpectedMap) {
        EXPECT_EQ(compute_unit_to_string(unit), str) 
            << "枚举转换失败: unit=" << static_cast<int>(unit);
    }

    // 测试字符串 → 枚举（正向映射）
    for (const auto& [unit, str] : kExpectedMap) {
        if (unit == ComputeUnit::UNKNOWN) continue;  // UNKNOWN无反向映射
        EXPECT_EQ(string_to_compute_unit(str), unit)
            << "字符串转换失败: str=" << str;
    }

    // 测试无效字符串 → 枚举（边界情况）
    EXPECT_EQ(string_to_compute_unit("invalid_unit"), ComputeUnit::UNKNOWN);
    EXPECT_EQ(string_to_compute_unit(""), ComputeUnit::UNKNOWN);
    EXPECT_EQ(string_to_compute_unit("CPU"), ComputeUnit::UNKNOWN);  // 区分大小写
}

// 测试2：DeviceId 字符串格式化
TEST(PlatformTest, DeviceIdToString) {
    // 正常情况
    DeviceId cpu_id(ComputeUnit::CPU, 0);
    EXPECT_EQ(cpu_id.to_string(), "cpu:0");

    DeviceId rknn_id(ComputeUnit::RK_NPU, 1);
    EXPECT_EQ(rknn_id.to_string(), "rk_npu:1");

    // 边界情况（UNKNOWN类型）
    DeviceId unknown_id(ComputeUnit::UNKNOWN, 99);
    EXPECT_EQ(unknown_id.to_string(), "unknown:99");
}

// 测试3：计算单元 → 芯片平台映射
TEST(PlatformTest, ChipPlatformMapping) {
    // 定义预期映射关系
    struct TestCase {
        ComputeUnit unit;
        ChipPlatform expected_platform;
    };
    const std::vector<TestCase> test_cases = {
        {ComputeUnit::CPU, ChipPlatform::GENERIC},
        {ComputeUnit::RK_NPU, ChipPlatform::ROCKCHIP},
        {ComputeUnit::RK_VPU, ChipPlatform::ROCKCHIP},
        {ComputeUnit::NVIDIA_CUDA, ChipPlatform::NVIDIA},
        {ComputeUnit::NVIDIA_DLA, ChipPlatform::NVIDIA},
        {ComputeUnit::ASCEND_NPU, ChipPlatform::ASCEND},
        {ComputeUnit::UNKNOWN, ChipPlatform::UNKNOWN}
    };

    // 验证每个计算单元对应的平台
    for (const auto& tc : test_cases) {
        EXPECT_EQ(get_chip_platform(tc.unit), tc.expected_platform)
            << "平台映射失败: unit=" << static_cast<int>(tc.unit);
    }
}

// 测试4：设备属性查询（核心业务逻辑）
TEST(PlatformTest, DevicePropertiesQuery) {
    // 测试CPU属性
    {
        DeviceId cpu_id(ComputeUnit::CPU, 0);
        auto prop = get_device_properties(cpu_id);
        
        EXPECT_EQ(prop.unit, ComputeUnit::CPU);
        EXPECT_EQ(prop.chip_platform, ChipPlatform::GENERIC);
        EXPECT_EQ(prop.chip_model, "Generic CPU");
        EXPECT_EQ(prop.memory_size, 8ULL * 1024 * 1024 * 1024);  // 8GB
        EXPECT_EQ(prop.max_batch_size, 16);
        EXPECT_TRUE(prop.supports_preprocess);
        EXPECT_TRUE(prop.supports_infer);
        EXPECT_TRUE(prop.supports_postprocess);
    }

    // 测试RK_NPU属性（模拟环境）
    {
        DeviceId rknn_id(ComputeUnit::RK_NPU, 0);
        auto prop = get_device_properties(rknn_id);
        
        EXPECT_EQ(prop.unit, ComputeUnit::RK_NPU);
        EXPECT_EQ(prop.chip_platform, ChipPlatform::ROCKCHIP);
        EXPECT_EQ(prop.chip_model, "RK (simulated)");  // 未定义RK_PLATFORM时的默认值
        EXPECT_EQ(prop.memory_size, 2ULL * 1024 * 1024 * 1024);  // 2GB
        EXPECT_EQ(prop.max_batch_size, 2);
        EXPECT_FALSE(prop.supports_preprocess);
        EXPECT_TRUE(prop.supports_infer);
        EXPECT_FALSE(prop.supports_postprocess);
    }

    // 测试NVIDIA_CUDA属性（模拟环境）
    {
        DeviceId cuda_id(ComputeUnit::NVIDIA_CUDA, 0);
        auto prop = get_device_properties(cuda_id);
        
        EXPECT_EQ(prop.unit, ComputeUnit::NVIDIA_CUDA);
        EXPECT_EQ(prop.chip_platform, ChipPlatform::NVIDIA);
        EXPECT_EQ(prop.chip_model, "NVIDIA (simulated)");  // 未定义NVIDIA_PLATFORM时的默认值
        EXPECT_EQ(prop.memory_size, 8ULL * 1024 * 1024 * 1024);  // 8GB
        EXPECT_EQ(prop.max_batch_size, 8);
        EXPECT_FALSE(prop.supports_preprocess);
        EXPECT_TRUE(prop.supports_infer);
        EXPECT_FALSE(prop.supports_postprocess);
    }

    // 测试未知设备属性（边界情况）
    {
        DeviceId unknown_id(ComputeUnit::UNKNOWN, 99);
        auto prop = get_device_properties(unknown_id);
        
        EXPECT_EQ(prop.unit, ComputeUnit::UNKNOWN);
        EXPECT_EQ(prop.chip_platform, ChipPlatform::UNKNOWN);
        EXPECT_EQ(prop.chip_model, "unknown");
        EXPECT_EQ(prop.memory_size, 0);
        EXPECT_EQ(prop.max_batch_size, 1);
        EXPECT_FALSE(prop.supports_preprocess);
        EXPECT_FALSE(prop.supports_infer);
        EXPECT_FALSE(prop.supports_postprocess);
    }
}

}  // namespace test
}  // namespace common
}  // namespace deploy