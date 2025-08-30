#include "perception_model/Yolov5SegmentationModel.hpp"

// 修改构造函数参数类型
Yolov5SegmentationModel::Yolov5SegmentationModel(const PerceptionConfig &config) : PerceptionModel(config)
{
    // 构造函数的具体实现
}

void Yolov5SegmentationModel::run(const PerceptionData &src, PerceptionResult &dst)
{
    const ImageInputData *imageInput = dynamic_cast<const ImageInputData *>(&src);
    Yolov5SegmentationResult *yolov5Result = dynamic_cast<Yolov5SegmentationResult *>(&dst);

    if (imageInput && yolov5Result)
    {
        // 运行模型的实现
        std::cout << "Running Yolov5SegmentationModel" << std::endl;
        // 这里可以添加针对 imageInput 和 yolov5Result 的具体处理逻辑
    }
    else
    {
        std::cerr << "Type conversion failed in Yolov5SegmentationModel::run" << std::endl;
    }
}