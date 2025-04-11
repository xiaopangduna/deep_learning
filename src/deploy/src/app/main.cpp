#include <iostream>
#include "configs/yaml_node_wrapper.h"
#include "perception_model/PerceptionModelFactory.hpp"
// #include "perception_model/PerceptionConfig.hpp"
// #include "perception_model/PerceptionModel.hpp"
// 
int main()
{
    std::cout << "Configuration content:\n" <<std::endl;
    Yolov5SegmentationModelConfig config;  // 创建派生类对象
    // 初始化 config

    PerceptionModelFactory::PerceptionModelType modelType = PerceptionModelFactory::PerceptionModelType::YOLOV5SEGMENTATION;
    // 直接传递指针
    PerceptionModel* model = PerceptionModelFactory::createPerceptionModel(modelType, &config);

    if (model) {
        model->run(); // 假设 PerceptionModel 有一个 run 方法
        delete model; // 释放模型资源
    } else {
        std::cerr << "Failed to create the perception model." << std::endl;
    }
    return 0;
}
