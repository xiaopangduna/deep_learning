#include "infer_engine/platform/rknn/RknnRunner.hpp"
#include <cstring>

namespace dp = deploy::perception::infer_engine::platform::rknn;
namespace types = deploy::perception::types;

dp::RknnRunner::RknnRunner() = default;
dp::RknnRunner::~RknnRunner() { Release(); }

bool dp::RknnRunner::Init(const std::string& model_path, std::string* err) {

    return true;
}

void dp::RknnRunner::Release() {

}

bool dp::RknnRunner::InferToTensors(const std::vector<types::TensorPtr>& /*inputs*/,
                                    std::vector<types::TensorPtr>& /*outputs*/,
                                    std::string* err) {
   
    return false;
}