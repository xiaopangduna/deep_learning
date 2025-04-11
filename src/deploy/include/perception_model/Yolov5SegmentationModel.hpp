#ifndef YOLOV5_SEGMENTATION_MODEL_HPP
#define YOLOV5_SEGMENTATION_MODEL_HPP

#include "perception_model/PerceptionModel.hpp"
#include "config/PerceptionConfig.hpp"

class Yolov5SegmentationModel : public PerceptionModel {
public:
    // 修改构造函数参数类型
    Yolov5SegmentationModel(const PerceptionConfig* config);
    // 重写基类的 loadModel 方法，实现加载模型的具体逻辑

    // 重写基类的 run 方法，实现运行模型的具体逻辑
    void run() override;
    ~Yolov5SegmentationModel() override = default; // 实现虚析构函数
};

#endif // YOLOV5_SEGMENTATION_MODEL_HPP
