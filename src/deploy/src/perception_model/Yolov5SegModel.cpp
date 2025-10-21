// #include "perception_model/Yolov5SegModel.hpp"
// #include "infer_engine/backend/rknn/RknnEngine.hpp"
// #include "preprocess/Yolov5SegPreprocess.hpp"
// #include "postprocess/Yolov5SegPostprocess.hpp"

// namespace perception {
// bool Yolov5SegModel::init(const Yolov5SegModelConfig& config) {
//     config_ = config;
//     // 1. 创建预处理实例
//     preprocessor_ = std::make_unique<preprocess::Yolov5SegPreprocess>(
//         config.input_h, config.input_w);

//     // 2. 创建后处理实例
//     postprocessor_ = std::make_unique<postprocess::Yolov5SegPostprocess>(
//         config.conf_thresh, config.nms_thresh);

//     // 3. 创建推理引擎（根据配置选择后端）
//     if (config.backend == "rknn") {
//         engine_ = std::make_unique<infer_engine::RknnEngine>();
//     } else if (config.backend == "onnxruntime") {
//         // 后续实现ONNX Runtime后端
//         return false;
//     } else {
//         return false;
//     }

//     // 4. 初始化推理引擎
//     infer_engine::Device device(infer_engine::DeviceType::NPU, config.device_id);
//     return engine_->init(config.model_path, device);
// }

// bool Yolov5SegModel::infer(const cv::Mat& src, SegmentationResult& result) {
//     // 1. 预处理
//     Tensor input_tensor;
//     float scale;
//     int pad_h, pad_w;
//     if (!preprocessor_->process(src, input_tensor, scale, pad_h, pad_w)) {
//         return false;
//     }

//     // 2. 推理
//     if (!engine_->set_input({input_tensor}) || !engine_->infer()) {
//         free(input_tensor.data);  // 释放预处理分配的内存
//         return false;
//     }
//     free(input_tensor.data);

//     // 3. 后处理
//     auto outputs = engine_->get_outputs();
//     result.img_h = src.rows;
//     result.img_w = src.cols;
//     return postprocessor_->process(outputs, scale, pad_h, pad_w, result);
// }
// }  // namespace perception