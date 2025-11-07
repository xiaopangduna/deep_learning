#pragma once
#include "infer_engine/Engine.hpp"
#include "infer_engine/platform/rknn/RknnRunner.hpp"
#include <memory>
#include <vector>

namespace deploy::perception::infer_engine::platform::rknn {

using TensorPtr = deploy::perception::types::TensorPtr;

class RknnEngine : public deploy::perception::infer_engine::Engine {
public:
    RknnEngine() = default;
    ~RknnEngine() override;

    bool Init(const std::string& model_path, std::string* err = nullptr) override;
    void Release() override;

    bool Infer(const std::vector<TensorPtr>& inputs,
               std::vector<TensorPtr>& outputs,
               std::string* err = nullptr) override;

private:
    std::unique_ptr<RknnRunner> runner_; // now directly holds the runner
};

} // namespace