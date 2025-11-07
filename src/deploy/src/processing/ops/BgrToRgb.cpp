#include "processing/ops/BgrToRgb.hpp"
#include "processing/ImageCvUtils.hpp"
#include <opencv2/opencv.hpp>

namespace dpp = deploy::perception::processing;
namespace types = deploy::perception::types;

namespace deploy::perception::processing {

bool BgrToRgbOp::Init(const Params& /*params*/, std::string* /*err*/) {
    return true;
}

bool BgrToRgbOp::Run(const TensorPtr& in, TensorPtr& out, std::string* err) const {
    cv::Mat src;
    if (!dpp::TensorToCvMat(in, src, err)) return false;
    if (src.empty()) { out.reset(); return true; }

    cv::Mat dst;
    if (src.channels() == 3) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, dst, cv::COLOR_BGRA2RGBA);
    } else {
        // single channel or other: just copy
        dst = src.clone();
    }

    out = dpp::CvMatToTensor(dst, in->dtype);
    return true;
}

} // namespace deploy::perception::processing