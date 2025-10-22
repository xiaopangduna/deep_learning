#include <gtest/gtest.h>
#include "common/Platform.hpp"
#include <string>
#include <vector>

using namespace deploy::common;

// 测试计算单元↔字符串转换（无日志）
TEST(PlatformAutoTest, ComputeUnitStringConversion) {
    EXPECT_EQ(compute_unit_to_string(ComputeUnit::CPU), "cpu");
    EXPECT_EQ(compute_unit_to_string(ComputeUnit::RK_NPU), "rk_npu");
    EXPECT_EQ(string_to_compute_unit("cpu"), ComputeUnit::CPU);
    EXPECT_EQ(string_to_compute_unit("invalid"), ComputeUnit::UNKNOWN);
}

// 测试DeviceId转换（无日志）
TEST(PlatformAutoTest, DeviceIdToString) {
    DeviceId cpu0(ComputeUnit::CPU, 0);
    EXPECT_EQ(cpu0.to_string(), "cpu:0");
    DeviceId rknn1(ComputeUnit::RK_NPU, 1);
    EXPECT_EQ(rknn1.to_string(), "rk_npu:1");
}

// 测试芯片平台映射（无日志）
TEST(PlatformAutoTest, ChipPlatformMapping) {
    EXPECT_EQ(get_chip_platform(ComputeUnit::RK_NPU), ChipPlatform::ROCKCHIP);
    EXPECT_EQ(get_chip_platform(ComputeUnit::CPU), ChipPlatform::GENERIC);
}

// 测试设备属性基础合法性（无日志）
TEST(PlatformAutoTest, DevicePropertiesBasicValidation) {
    DeviceId cpu_id(ComputeUnit::CPU, 0);
    DeviceProperties cpu_prop = get_device_properties(cpu_id);
    EXPECT_EQ(cpu_prop.unit, ComputeUnit::CPU);
    EXPECT_GT(cpu_prop.memory_size, 0);
    EXPECT_TRUE(cpu_prop.supports_infer);
}

// 测试RK NPU属性（无日志）
TEST(PlatformAutoTest, RKNpuPropertiesValidation) {
#ifdef RK_PLATFORM
    DeviceId rknn_id(ComputeUnit::RK_NPU, 0);
    DeviceProperties rk_prop = get_device_properties(rknn_id);
    EXPECT_EQ(rk_prop.unit, ComputeUnit::RK_NPU);
    EXPECT_NE(rk_prop.chip_model, "unknown");
    EXPECT_TRUE(rk_prop.supports_infer);
#else
    GTEST_SKIP();
#endif
}