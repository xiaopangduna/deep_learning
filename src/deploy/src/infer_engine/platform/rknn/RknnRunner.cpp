#include "infer_engine/platform/rknn/RknnRunner.hpp"
#ifdef ENABLE_RKNN
#include "rknn_api.h"
#endif
#include <cstring>

namespace dp = deploy::perception::infer_engine::platform::rknn;
namespace types = deploy::perception::types;

dp::RknnRunner::RknnRunner() = default;
dp::RknnRunner::~RknnRunner() { Release(); }

bool dp::RknnRunner::Init(const std::string& model_path, std::string* err) {
#ifdef ENABLE_RKNN
    // 使用真实 RKNN API
    rknn_context ctx = 0;
    // ... 调用 rknn_init / rknn_query 等，并把 ctx 存到 ctx_（reinterpret_cast）
    ctx_ = reinterpret_cast<void*>(ctx);
    return true;
#else
    // stub 行为，允许项目在无 SDK 环境下编译
    (void)model_path; (void)err;
    ctx_ = nullptr;
    return true;
#endif
}

void dp::RknnRunner::Release() {
#ifdef ENABLE_RKNN
    if (ctx_) {
        rknn_destroy(reinterpret_cast<rknn_context>(ctx_));
        ctx_ = nullptr;
    }
#else
    ctx_ = nullptr;
#endif
    input_attrs_.clear();
    output_attrs_.clear();
    io_num_ = {};
    is_quant_ = false;
}

bool dp::RknnRunner::InferToTensors(const std::vector<types::TensorPtr>& /*inputs*/,
                                    std::vector<types::TensorPtr>& /*outputs*/,
                                    std::string* err) {
    if (err) *err = "RknnRunner::InferToTensors: stub implementation (not yet implemented)";
    return false;
}