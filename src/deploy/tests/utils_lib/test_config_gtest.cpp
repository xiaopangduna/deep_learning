#include <gtest/gtest.h>

#include "../../lib/utils_lib/include/configs/yaml_config.h"



// 测试 YamlConfig 类的加载功能
TEST(YamlConfigTest, LoadConfig) {
    YamlConfig yamlConfig;
    // 假设存在一个测试用的 YAML 文件
    bool result = yamlConfig.load("/workspace/deploy/tests/utils_lib/test_yaml.yaml");
    EXPECT_TRUE(result) << "Failed to load YAML config file";
}

// 测试 YamlConfig 类获取值的功能
TEST(YamlConfigTest, GetValue) {
    YamlConfig yamlConfig;
    if (yamlConfig.load("/workspace/deploy/tests/utils_lib/test_yaml.yaml")) {
        std::string value = yamlConfig.get("example_key");
        EXPECT_FALSE(value.empty()) << "Failed to get value from YAML config";
    } else {
        FAIL() << "Could not load YAML config for GetValue test";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
