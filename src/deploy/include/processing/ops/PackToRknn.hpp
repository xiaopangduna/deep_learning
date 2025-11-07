#pragma once
#include "processing/Op.hpp"
#include <string>

namespace deploy::perception::processing {

class PackToRknnOp : public Op {
public:
    PackToRknnOp() = default;
    ~PackToRknnOp() override = default;

    bool Init(const Params& params, std::string* err = nullptr) override;
    // in: preprocessed tensor (HWC or {1,C,H,W}). out: contiguous tensor matching model (UINT8 NHWC).
    bool Run(const TensorPtr& in, TensorPtr& out, std::string* err = nullptr) const override;
    std::string name() const noexcept override { return "PackToRknn"; }

private:
    std::string model_fmt_ = "UINT8_NHWC"; // currently only this supported
};

} // namespace deploy::perception::processing