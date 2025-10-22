// #ifndef INFER_ENGINE_RKNN_ENGINE_HPP
// #define INFER_ENGINE_RKNN_ENGINE_HPP

// #include "infer_engine/InferenceEngine.hpp"
// // #include "rknn_api.h"

// namespace infer_engine {
// class RknnEngine : public InferenceEngine {
// public:
//     ~RknnEngine() override;
//     bool init(const std::string& model_path, const Device& device) override;
//     bool set_input(const std::vector<Tensor>& inputs) override;
//     bool infer() override;
//     std::vector<Tensor> get_outputs() override;
//     const ModelInfo& get_model_info() const override { return model_info_; }

// private:
//     rknn_context ctx_ = 0;  // RKNN上下文
//     ModelInfo model_info_;  // 模型元信息
//     std::vector<rknn_tensor_attr> input_attrs_;  // 输入属性
//     std::vector<rknn_tensor_attr> output_attrs_;  // 输出属性
//     std::vector<void*> input_buffers_;  // 输入缓冲区（对齐RKNN要求）
//     std::vector<void*> output_buffers_;  // 输出缓冲区
// };
// }  // namespace infer_engine

// #endif  // INFER_ENGINE_RKNN_ENGINE_HPP