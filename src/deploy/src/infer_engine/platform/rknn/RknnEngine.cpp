#include "infer_engine/platform/rknn/RknnEngine.hpp"

namespace dpeng = deploy::perception::infer_engine::platform::rknn;

dpeng::RknnEngine::~RknnEngine() {
    Release();
}

bool dpeng::RknnEngine::Init(const std::string& model_path, std::string* err) {

}

void dpeng::RknnEngine::Release() {

}

bool dpeng::RknnEngine::Infer(const std::vector<TensorPtr>& inputs,
                              std::vector<TensorPtr>& outputs,
                              std::string* err) {

}