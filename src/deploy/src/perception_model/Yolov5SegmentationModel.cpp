#include "perception_model/Yolov5SegmentationModel.hpp"

// 修改构造函数参数类型
Yolov5SegmentationModel::Yolov5SegmentationModel(const PerceptionConfig* config) : PerceptionModel(config) {
    // 构造函数实现
}

void Yolov5SegmentationModel::run() {
    // 运行模型的实现
    std::cout << "Running Yolov5SegmentationModel" << std::endl;
}