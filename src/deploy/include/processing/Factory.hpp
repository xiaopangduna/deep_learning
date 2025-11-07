#pragma once
#include <memory>
#include <string>
#include <vector>
#include "type/Step.hpp"
#include "processing/PreprocessPipeline.hpp"
// #include "processing/PostprocessPipeline.hpp"


namespace deploy::perception::processing {

// Build a preprocess pipeline from cfg.pre.steps.
// On failure returns nullptr and writes a human-readable reason to err (if non-null).
std::unique_ptr<PreprocessPipeline> BuildPreprocessPipeline(const std::vector<deploy::perception::types::Step>& steps,
                                                            std::string* err = nullptr);

// // Build a postprocess pipeline from cfg.post.steps.
// // On failure returns nullptr and writes a human-readable reason to err (if non-null).
// std::unique_ptr<PostprocessPipeline> BuildPostprocessPipeline(const std::vector<Step>& steps,
//                                                               std::string* err = nullptr);

} // namespace deploy::perception::processing