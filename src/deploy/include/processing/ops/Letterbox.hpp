#pragma once
#include "processing/Op.hpp"
#include <string>

namespace deploy::perception::processing {

class LetterboxOp : public Op {
public:
    LetterboxOp() = default;
    ~LetterboxOp() override = default;

    bool Init(const Params& params, std::string* err = nullptr) override;
    bool Run(const TensorPtr& in, TensorPtr& out, std::string* err = nullptr) const override;
    std::string name() const noexcept override { return "Letterbox"; }

private:
    int target_w_ = 0;
    int target_h_ = 0;
    int bg_color_ = 114;
    std::string mode_ = "letterbox"; // reserved (could support stretch)
};

} // namespace deploy::perception::processing