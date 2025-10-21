#include "infer_engine/backend/rknn/RknnEngine.hpp"
#include <fstream>
#include <cstring>
#include <posix_memalign.h>

namespace infer_engine {
RknnEngine::~RknnEngine() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
    }
    for (auto buf : input_buffers_) free(buf);
    for (auto buf : output_buffers_) free(buf);
}

bool RknnEngine::init(const std::string& model_path, const Device& device) {
    // 1. 加载RKNN模型文件
    std::ifstream fp(model_path, std::ios::binary);
    if (!fp) return false;
    fp.seekg(0, std::ios::end);
    size_t model_size = fp.tellg();
    fp.seekg(0, std::ios::beg);
    char* model_data = new char[model_size];
    fp.read(model_data, model_size);
    fp.close();

    // 2. 初始化RKNN上下文
    int ret = rknn_init(&ctx_, model_data, model_size, 0, nullptr);
    delete[] model_data;
    if (ret != RKNN_SUCC) return false;

    // 3. 获取输入输出属性并填充model_info_
    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return false;

    // 解析输入信息
    input_attrs_.resize(io_num.n_input);
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, input_attrs_.data(), 
                   sizeof(rknn_tensor_attr) * io_num.n_input);
    if (ret != RKNN_SUCC) return false;
    for (auto& attr : input_attrs_) {
        TensorInfo info;
        info.name = "input_" + std::to_string(attr.index);
        info.shape = {attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3]};  // NCHW
        info.dtype = (attr.data_type == RKNN_TENSOR_FLOAT32) ? DataType::FP32 : DataType::INT8;
        model_info_.add_input(info);
    }

    // 解析输出信息
    output_attrs_.resize(io_num.n_output);
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, output_attrs_.data(), 
                   sizeof(rknn_tensor_attr) * io_num.n_output);
    if (ret != RKNN_SUCC) return false;
    for (auto& attr : output_attrs_) {
        TensorInfo info;
        info.name = "output_" + std::to_string(attr.index);
        info.shape = {attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3]};
        info.dtype = (attr.data_type == RKNN_TENSOR_FLOAT32) ? DataType::FP32 : DataType::INT8;
        model_info_.add_output(info);
    }

    // 4. 分配输入输出缓冲区（16字节对齐）
    for (auto& attr : input_attrs_) {
        void* buf;
        posix_memalign(&buf, 16, attr.size);
        input_buffers_.push_back(buf);
    }
    for (auto& attr : output_attrs_) {
        void* buf;
        posix_memalign(&buf, 16, attr.size);
        output_buffers_.push_back(buf);
    }

    return true;
}

bool RknnEngine::set_input(const std::vector<Tensor>& inputs) {
    if (inputs.size() != input_buffers_.size()) return false;
    for (size_t i = 0; i < inputs.size(); ++i) {
        // 检查输入尺寸是否匹配
        if (inputs[i].total_bytes() != input_attrs_[i].size) return false;
        // 复制数据到对齐缓冲区
        memcpy(input_buffers_[i], inputs[i].data, inputs[i].total_bytes());
    }
    return true;
}

bool RknnEngine::infer() {
    // 设置输入
    std::vector<rknn_input> rknn_inputs(input_attrs_.size());
    for (size_t i = 0; i < input_attrs_.size(); ++i) {
        rknn_inputs[i].index = input_attrs_[i].index;
        rknn_inputs[i].buf = input_buffers_[i];
        rknn_inputs[i].size = input_attrs_[i].size;
        rknn_inputs[i].type = (input_attrs_[i].data_type == RKNN_TENSOR_FLOAT32) ? RKNN_TENSOR_FLOAT32 : RKNN_TENSOR_INT8;
        rknn_inputs[i].pass_through = 0;
    }
    int ret = rknn_inputs_set(ctx_, input_attrs_.size(), rknn_inputs.data());
    if (ret != RKNN_SUCC) return false;

    // 执行推理
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) return false;

    // 获取输出
    std::vector<rknn_output> rknn_outputs(output_attrs_.size());
    for (size_t i = 0; i < output_attrs_.size(); ++i) {
        rknn_outputs[i].index = output_attrs_[i].index;
        rknn_outputs[i].want_float = 1;  // 要求输出FP32
        rknn_outputs[i].buf = output_buffers_[i];
        rknn_outputs[i].size = output_attrs_[i].size;
    }
    ret = rknn_outputs_get(ctx_, output_attrs_.size(), rknn_outputs.data(), 0);
    return ret == RKNN_SUCC;
}

std::vector<Tensor> RknnEngine::get_outputs() {
    std::vector<Tensor> outputs;
    for (size_t i = 0; i < output_attrs_.size(); ++i) {
        Tensor tensor;
        tensor.data = output_buffers_[i];
        tensor.shape = model_info_.outputs()[i].shape;
        tensor.dtype = model_info_.outputs()[i].dtype;
        outputs.push_back(tensor);
    }
    return outputs;
}
}  // namespace infer_engine