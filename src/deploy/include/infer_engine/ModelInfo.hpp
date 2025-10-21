#ifndef INFER_ENGINE_MODEL_INFO_HPP
#define INFER_ENGINE_MODEL_INFO_HPP

#include <vector>
#include "type/PerceptionData.hpp"  // 依赖TensorInfo定义

namespace infer_engine {
// 模型输入/输出张量的元信息
struct TensorInfo {
    std::string name;  // 张量名称（如"images"）
    std::vector<int> shape;  // 形状（如[1,3,640,640]）
    DataType dtype;  // 数据类型
};

// 模型整体元信息（输入/输出描述）
class ModelInfo {
public:
    void add_input(const TensorInfo& input);
    void add_output(const TensorInfo& output);
    const std::vector<TensorInfo>& inputs() const;
    const std::vector<TensorInfo>& outputs() const;
    // 检查输入形状是否符合预期（辅助功能）
    bool check_input_shape(int input_idx, const std::vector<int>& expected_shape) const;

private:
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
};
}  // namespace infer_engine

#endif  // INFER_ENGINE_MODEL_INFO_HPP