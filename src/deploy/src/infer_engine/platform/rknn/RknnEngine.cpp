#include "infer_engine/platform/rknn/RknnEngine.hpp"

namespace dpeng = deploy::perception::infer_engine::platform::rknn;

dpeng::RknnEngine::~RknnEngine() {
    Release();
}

bool dpeng::RknnEngine::Init(const std::string& model_path, std::string* err) {
    runner_.reset(new RknnRunner());
    if (!runner_) {
        if (err) *err = "failed to allocate RknnRunner";
        return false;
    }
    if (!runner_->Init(model_path, err)) {
        runner_.reset();
        return false;
    }
    return true;
}

void dpeng::RknnEngine::Release() {
    if (runner_) {
        runner_->Release();
        runner_.reset();
    }
}

bool dpeng::RknnEngine::Infer(const std::vector<TensorPtr>& inputs,
                              std::vector<TensorPtr>& outputs,
                              std::string* err) {
    if (!runner_) { if (err) *err = "RknnEngine not initialized"; return false; }
    return runner_->InferToTensors(inputs, outputs, err);
}