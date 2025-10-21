#include "infer_engine/ModelInfo.hpp"
#include <cassert>

namespace infer_engine {
void ModelInfo::add_input(const TensorInfo& input) {
    inputs_.push_back(input);
}

void ModelInfo::add_output(const TensorInfo& output) {
    outputs_.push_back(output);
}

const std::vector<TensorInfo>& ModelInfo::inputs() const {
    return inputs_;
}

const std::vector<TensorInfo>& ModelInfo::outputs() const {
    return outputs_;
}

// 检查输入形状是否匹配（例如：是否为NCHW格式）
bool ModelInfo::check_input_shape(int input_idx, const std::vector<int>& expected_shape) const {
    if (input_idx >= inputs_.size()) return false;
    const auto& actual_shape = inputs_[input_idx].shape;
    return actual_shape == expected_shape;
}
}  // namespace infer_engine