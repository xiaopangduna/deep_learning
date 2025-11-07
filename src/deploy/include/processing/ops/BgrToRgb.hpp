#pragma once
#include "processing/Op.hpp"
#include <string>

namespace deploy::perception::processing {

class BgrToRgbOp : public Op {
public:
    BgrToRgbOp() = default;
    ~BgrToRgbOp() override = default;

    bool Init(const Params& params, std::string* err = nullptr) override;
    bool Run(const TensorPtr& in, TensorPtr& out, std::string* err = nullptr) const override;
    std::string name() const noexcept override { return "BgrToRgb"; }
};

} // namespace deploy::perception::processing