#include "infer_engine/Engine.hpp"
#include <memory>
#include <string>

#ifdef ENABLE_RKNN
#include "infer_engine/platform/rknn/RknnEngine.hpp"
#endif

namespace dp = deploy::perception::infer_engine;

dp::EnginePtr CreateEngine(const std::string& backend) {
    if (backend == "rknn") {
#ifdef ENABLE_RKNN
        return std::make_unique<deploy::perception::infer_engine::platform::rknn::RknnEngine>();
#else
        return nullptr;
#endif
    }
    if (backend == "onnxruntime") {
#ifdef ENABLE_ONNXRUNTIME
        return std::make_unique<deploy::perception::infer_engine::platform::onnxruntime::OnnxRuntimeEngine>();
#else
        return nullptr;
#endif
    }
    return nullptr;
}