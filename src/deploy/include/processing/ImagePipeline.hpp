#pragma once

#include "processing/Pipeline.hpp"

namespace deploy {
namespace processing {

// ImagePipeline is a semantic alias for pipeline that converts images -> tensors.
// Additional helpers / builders can be added here later.
class ImagePipeline : public Pipeline {
public:
    ImagePipeline() = default;
    ~ImagePipeline() = default;
};

} // namespace processing
} // namespace deploy