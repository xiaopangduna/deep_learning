#pragma once
#include <memory>
#include <vector>
#include <string>
#include "type/Tensor.hpp"

namespace deploy::perception::processing {

using TensorPtr = std::shared_ptr<deploy::perception::types::Tensor>;

// 预处理流水线抽象：批量接口，输入/输出为 TensorPtr 向量。
// - in/out 长度一一对应；失败项可用空的 TensorPtr 占位。
// - per_tensor_err 可选，用于返回每个输入的错误信息（与 in 顺序对应）。
class PreprocessPipeline {
public:
    virtual ~PreprocessPipeline() = default;

    // 单一批量入口：输入 tensors -> 输出 tensors
    virtual bool Run(const std::vector<TensorPtr>& in,
                     std::vector<TensorPtr>& out,
                     std::vector<std::string>* per_tensor_err = nullptr) const = 0;
};

using PreprocessPipelinePtr = std::unique_ptr<PreprocessPipeline>;

} // namespace deploy::perception::processing