#pragma once
#include <memory>
#include <string>
#include <vector>
#include "processing/ModelOutputs.hpp"
#include "processing/ImageMeta.hpp"
#include "processing/Step.hpp"

namespace deploy::perception::processing {

// 后处理流水线抽象：将模型原始输出和对应的 ImageMeta 转换为最终结果。
// ResultType 在此以字符串占位（实际项目中替换为结构化结果类型）。
class PostprocessPipeline {
public:
    virtual ~PostprocessPipeline() = default;

    // 处理单个样本的原始模型输出，返回 true 表示成功并在 out_result 填入文本化结果（占位）。
    virtual bool Run(const ModelOutputs& raw, const ImageMeta& meta, std::string* out_result,
                     std::string* err = nullptr) const = 0;

    // 批量后处理； raws 与 metas 长度应一致； per_image_err 可选填充每张的错误信息。
    virtual bool RunBatch(const std::vector<ModelOutputs>& raws,
                          const std::vector<ImageMeta>& metas,
                          std::vector<std::string>* out_results,
                          std::vector<std::string>* per_image_err = nullptr) const = 0;
};

using PostprocessPipelinePtr = std::unique_ptr<PostprocessPipeline>;

} // namespace deploy::perception::processing