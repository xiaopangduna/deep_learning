// #ifndef PERCEPTION_MODEL_FACTORY_HPP
// #define PERCEPTION_MODEL_FACTORY_HPP

// #include <string>
// #include <stdexcept>
// #include "config/PerceptionConfig.hpp"
// #include "perception_model/PerceptionModel.hpp"
// #include "perception_model/Yolov5SegModel.hpp"

// class PerceptionModelFactory {
// public:
//     // 定义 PerceptionModelType 枚举
//     enum class PerceptionModelType {
//         YOLOV5SEGMENTATION
//         // 可以添加其他模型类型
//     };
// private:
//     const PerceptionConfig* config;
// public:
//     PerceptionModelFactory(const PerceptionConfig* config) : config(config) {}
//     static PerceptionModel* createPerceptionModel(PerceptionModelType modelType, const PerceptionConfig& config);
// };

// #endif // PERCEPTION_MODEL_FACTORY_HPP
