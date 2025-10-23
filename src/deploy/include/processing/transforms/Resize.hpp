#pragma once
#include "processing/Transform.hpp"
#include <string>

namespace deploy {
namespace processing {
namespace transforms {

class ResizeTransform : public Transform {
public:
    // interp: "nearest" or "bilinear" (bilinear not implemented now), keep_aspect unused for now
    ResizeTransform(int target_w, int target_h, const std::string& interp = "nearest", bool keep_aspect = false)
        : target_w_(target_w), target_h_(target_h), interp_(interp), keep_aspect_(keep_aspect) {}

    bool apply(const Buffer& in, Buffer& out) override;
    std::string name() const override { return "Resize"; }

private:
    int target_w_;
    int target_h_;
    std::string interp_;
    bool keep_aspect_;
};

} // namespace transforms
} // namespace processing
} // namespace deploy