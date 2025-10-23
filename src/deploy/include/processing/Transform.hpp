#pragma once

#include "common/Tensor.hpp"
#include <string>
#include <memory>

namespace deploy {
namespace processing {

using Buffer = deploy::common::TensorBuffer;

class Transform {
public:
    virtual ~Transform() = default;
    // Apply transform: read from `in`, write to `out`. Return true on success.
    virtual bool apply(const Buffer& in, Buffer& out) = 0;
    virtual std::string name() const = 0;
};

using TransformPtr = std::unique_ptr<Transform>;

} // namespace processing
} // namespace deploy