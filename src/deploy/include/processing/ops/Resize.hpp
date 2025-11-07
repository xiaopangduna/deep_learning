#pragma once
#include "processing/Op.hpp"
#include <string>

namespace deploy::perception::processing {

class ResizeOp : public Op {
public:
    ResizeOp() = default;
    ~ResizeOp() override = default;

    bool Init(const Params& params, std::string* err = nullptr) override;
    bool Run(const TensorPtr& in, TensorPtr& out, std::string* err = nullptr) const override;
    std::string name() const noexcept override { return "Resize"; }

private:
    int target_w_ = 0;
    int target_h_ = 0;
    std::string mode_ = "stretch"; // reserved for future
};

} // namespace deploy::perception::processing