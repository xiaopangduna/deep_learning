#pragma once
#include <memory>
#include <string>
#include "infer_engine/Engine.hpp"

namespace deploy::perception::infer_engine {

// Create an Engine instance for the given backend string ("rknn", "onnxruntime", ...).
// Returns nullptr if the backend is unavailable or not enabled at build time.
EnginePtr CreateEngine(const std::string& backend);

} // namespace deploy::perception::infer_engine