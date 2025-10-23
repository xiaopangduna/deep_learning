#pragma once

#include "processing/Pipeline.hpp"

namespace deploy {
namespace processing {

// PostprocessPipeline is a semantic alias for pipeline that converts model outputs -> results.
class PostprocessPipeline : public Pipeline {
public:
    PostprocessPipeline() = default;
    ~PostprocessPipeline() = default;
};

} // namespace processing
} // namespace deploy