#pragma once
#include <memory>
#include <string>
#include <vector>
#include "type/Tensor.hpp"

namespace deploy::perception::infer_engine {

using TensorPtr = deploy::perception::types::TensorPtr;

// Abstract engine interface: inputs/outputs are arrays of Tensors.
// Engine only does inference (no postprocess).
class Engine {
public:
    virtual ~Engine() = default;

    // Initialize engine with model_path. Fill err on failure.
    virtual bool Init(const std::string& model_path, std::string* err = nullptr) = 0;

    // Release resources.
    virtual void Release() = 0;

    // Synchronous inference: inputs -> outputs. Caller owns inputs; outputs returned as host Tensors.
    // out vector will be filled with one Tensor per model output.
    virtual bool Infer(const std::vector<TensorPtr>& inputs,
                       std::vector<TensorPtr>& outputs,
                       std::string* err = nullptr) = 0;
};

using EnginePtr = std::unique_ptr<Engine>;
} // namespace deploy::perception::infer_engine