#include "perception_model/PerceptionModelFactory.hpp"

PerceptionModel* PerceptionModelFactory::createPerceptionModel(PerceptionModelFactory::PerceptionModelType modelType, const PerceptionConfig& config) {
    switch (modelType) {
        case PerceptionModelFactory::PerceptionModelType::YOLOV5SEGMENTATION:
            return new Yolov5SegmentationModel(config);
        default:
            return nullptr;
    }
}
