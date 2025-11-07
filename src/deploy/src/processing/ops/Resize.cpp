#include "processing/ops/Resize.hpp"
#include "processing/ImageCvUtils.hpp"
#include <opencv2/opencv.hpp>
#include <cstring>
#include <stdexcept>

namespace dpp = deploy::perception::processing;
namespace types = deploy::perception::types;

namespace deploy::perception::processing {

bool ResizeOp::Init(const Params& params, std::string* err) {
    try {
        if (params.count("width")) target_w_ = std::stoi(params.at("width"));
        if (params.count("height")) target_h_ = std::stoi(params.at("height"));
        if (params.count("mode")) mode_ = params.at("mode");
    } catch (...) {
        if (err) *err = "ResizeOp: invalid width/height";
        return false;
    }
    if (target_w_ <= 0 || target_h_ <= 0) {
        if (err) *err = "ResizeOp: width/height must be >0";
        return false;
    }
    return true;
}

bool ResizeOp::Run(const TensorPtr& in, TensorPtr& out, std::string* err) const {
    cv::Mat in_mat;
    if (!dpp::TensorToCvMat(in, in_mat, err)) return false;
    if (in_mat.empty()) {
        out.reset();
        return true;
    }
    cv::Mat resized;
    cv::resize(in_mat, resized, cv::Size(target_w_, target_h_));
    types::DType dt = (in->dtype == types::DType::FLOAT32) ? types::DType::FLOAT32 : types::DType::UINT8;
    out = dpp::CvMatToTensor(resized, dt);
    return true;
}

// register factory at load time
namespace {
bool reg = RegisterOpFactory("Resize", [](const Params& p, std::string* err)->OpPtr {
    auto r = std::make_unique<ResizeOp>();
    if (!r->Init(p, err)) return nullptr;
    return r;
});
}

} // namespace deploy::perception::processing