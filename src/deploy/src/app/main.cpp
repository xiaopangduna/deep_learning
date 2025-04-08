#include <iostream>
#include "configs/yaml_node_wrapper.h"

int main()
{
    // 使用命名空间限定符访问 YamlConfig 类
    std::string path_config = "/workspace/deploy/configs/yolov5_seg_fall_detection.yaml";
    LovelyUtils::YamlNodeWrapper yaml_config(path_config);

    // 设置新的值
    yaml_config.setValue("name", "Alice");
    yaml_config.setValue("age", 30);

    // 保存到文件
    if (yaml_config.saveToFile("config.yaml"))
    {
        std::cout << "Configuration saved successfully." << std::endl;
    }
    else
    {
        std::cerr << "Failed to save configuration." << std::endl;
    }
    std::string value = yaml_config.getNode()["parent"]["child"]["grandchild"].as<std::string>();
    std::cout << "Value: " << value << std::endl;
    // 打印当前配置
    std::string configStr = yaml_config.printToString();
    std::cout << "Configuration content:\n" << configStr;
    return 0;
}
