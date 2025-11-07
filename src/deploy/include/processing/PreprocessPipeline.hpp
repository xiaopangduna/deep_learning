#pragma once
#include <memory>
#include <vector>
#include <string>
#include "type/Tensor.hpp"

namespace deploy::perception::processing {

using TensorPtr = std::shared_ptr<deploy::perception::types::Tensor>;

// 预处理流水线抽象：批量接口，输入/输出为 TensorPtr 向量。
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