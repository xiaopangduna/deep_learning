#include <iostream>
#include "configs/yaml_node_wrapper.h"
#include "perception_model/PerceptionModelFactory.hpp"
#include "config/Yolov5SegmentationModelConfig.hpp"
#include "type/PerceptionData.hpp"


void printYamlNode(const YAML::Node& node, int depth = 0) {
    std::string indent(depth * 2, ' ');
    switch (node.Type()) {
        case YAML::NodeType::Scalar:
            std::cout << indent << node.as<std::string>() << std::endl;
            break;
        case YAML::NodeType::Sequence:
            for (size_t i = 0; i < node.size(); ++i) {
                std::cout << indent << "- ";
                printYamlNode(node[i], depth + 1);
            }
            break;
        case YAML::NodeType::Map:
            for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
                std::cout << indent << it->first.as<std::string>() << ": ";
                printYamlNode(it->second, depth + 1);
            }
            break;
        default:
            break;
    }
}
int main()
{
    std::cout << "Configuration content:\n" <<std::endl;
    Yolov5SegmentationModelConfig config;  // 创建派生类对象
    PerceptionModel* model = nullptr; // 初始化模型指针
    PerceptionModelFactory::PerceptionModelType modelType = PerceptionModelFactory::PerceptionModelType::YOLOV5SEGMENTATION;
    try {
        // 读取 YAML 文件
        YAML::Node yamlNode = YAML::LoadFile("../configs/yolov5_seg_fall_detection.yaml");
        // 遍历并打印 YAML 节点内容
        // printYamlNode(yamlNode);
 
        // 假设 Yolov5SegmentationModelConfig 有一个 loadFromYaml 方法来解析 YAML 数据
        config.loadFromYamlNode(yamlNode["fall_detection"]["yolov5Segmentation"]);
        config.printConfig();
        std::cout << "Parsed configuration:\n" << std::endl;
        
        model = PerceptionModelFactory::createPerceptionModel(modelType, config);
        // model->run(config, config); // 运行模型

    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return 1;
    }


    // if (model) {
    //     model->run(); // 假设 PerceptionModel 有一个 run 方法
    //     delete model; // 释放模型资源
    // } else {
    //     std::cerr << "Failed to create the perception model." << std::endl;
    // }
    return 0;
}
