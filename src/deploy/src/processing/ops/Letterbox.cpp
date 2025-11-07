#include "processing/ops/Letterbox.hpp"
#include "processing/ImageCvUtils.hpp"
#include <opencv2/opencv.hpp>
#include <cstring>

namespace dpp = deploy::perception::processing;
namespace types = deploy::perception::types;

namespace deploy::perception::processing {

bool LetterboxOp::Init(const Params& params, std::string* err) {
    try {
        if (params.count("width")) target_w_ = std::stoi(params.at("width"));
        if (params.count("height")) target_h_ = std::stoi(params.at("height"));
        if (params.count("bg_color")) bg_color_ = std::stoi(params.at("bg_color"));
        if (params.count("mode")) mode_ = params.at("mode");
    } catch (...) {
        if (err) *err = "LetterboxOp: invalid params";
        return false;
    }
    if (target_w_ <= 0 || target_h_ <= 0) {
        if (err) *err = "LetterboxOp: width/height must be >0";
        return false;
    }
    return true;
}

bool LetterboxOp::Run(const TensorPtr& in, TensorPtr& out, std::string* err) const {
    cv::Mat src;
    if (!dpp::TensorToCvMat(in, src, err)) return false;
    if (src.empty()) { out.reset(); return true; }

    // ensure 3-channel BGR for letterbox (like original demo)
    cv::Mat src3;
    if (src.channels() == 1) cv::cvtColor(src, src3, cv::COLOR_GRAY2BGR);
    else if (src.channels() == 4) cv::cvtColor(src, src3, cv::COLOR_BGRA2BGR);
    else src3 = src;

    int src_w = src3.cols, src_h = src3.rows;
    float scale = std::min(static_cast<float>(target_w_) / src_w, static_cast<float>(target_h_) / src_h);
    int new_w = static_cast<int>(round(src_w * scale));
    int new_h = static_cast<int>(round(src_h * scale));
    int pad_w = target_w_ - new_w;
    int pad_h = target_h_ - new_h;
    int pad_left = pad_w / 2;
    int pad_top = pad_h / 2;

    cv::Mat resized;
    cv::resize(src3, resized, cv::Size(new_w, new_h));

    cv::Mat dst(target_h_, target_w_, CV_8UC3, cv::Scalar(bg_color_, bg_color_, bg_color_));
    resized.copyTo(dst(cv::Rect(pad_left, pad_top, new_w, new_h)));

    // preserve dtype of input tensor
    types::DType out_dt = in->dtype;
    if (out_dt == types::DType::FLOAT32) {
        cv::Mat dstf;
        dst.convertTo(dstf, CV_32F, 1.0f); // no normalization here
        out = dpp::CvMatToTensor(dstf, types::DType::FLOAT32);
    } else {
        out = dpp::CvMatToTensor(dst, types::DType::UINT8);
    }
    return true;
}

} // namespace deploy::perception::processing