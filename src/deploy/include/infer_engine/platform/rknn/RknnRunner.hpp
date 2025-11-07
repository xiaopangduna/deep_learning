#pragma once
#include <string>
#include <vector>
#include "type/Tensor.hpp"

namespace deploy::perception::infer_engine::platform::rknn {

using TensorPtr = deploy::perception::types::TensorPtr;

class RknnRunner {
public:
    RknnRunner();
    ~RknnRunner();

    bool Init(const std::string& model_path, std::string* err = nullptr);
    void Release();
    bool InferToTensors(const std::vector<TensorPtr>& inputs,
                        std::vector<TensorPtr>& outputs,
                        std::string* err = nullptr);

    void* ctx() const { return ctx_; }

private:
    // opaque pointers hide SDK headers from consumers
    void* ctx_ = nullptr;
    void* io_num_ = nullptr;
    void* input_attrs_ = nullptr;
    void* output_attrs_ = nullptr;
    bool is_quant_ = false;
};

} // namespace