#include "processing/ops/Resize.hpp"
#include <opencv2/opencv.hpp>
#include <cstring>
#include <stdexcept>

namespace dpp = deploy::perception::processing;
namespace types = deploy::perception::types;

static bool TensorToCvMat(const dpp::TensorPtr& t, cv::Mat& out, std::string* err) {
    if (!t || !t->data) {
        if (err) *err = "TensorToCvMat: null tensor or data";
        return false;
    }
    if (t->device != "cpu") {
        if (err) *err = "TensorToCvMat: only cpu tensors supported in this example";
        return false;
    }
    if (t->shape.size() != 3) {
        if (err) *err = "TensorToCvMat: expected shape {H,W,C}";
        return false;
    }
    int H = t->shape[0], W = t->shape[1], C = t->shape[2];
    int cv_type = (t->dtype == types::DType::UINT8) ? CV_8UC(C) :
                  (t->dtype == types::DType::FLOAT32) ? CV_32FC(C) : -1;
    if (cv_type < 0) {
        if (err) *err = "TensorToCvMat: unsupported dtype";
        return false;
    }
    // Create Mat and copy bytes
    out = cv::Mat(H, W, cv_type);
    std::size_t copy_bytes = std::min<std::size_t>(out.total() * out.elemSize(), t->byte_size);
    std::memcpy(out.data, t->data, copy_bytes);
    return true;
}

static dpp::TensorPtr CvMatToTensor(const cv::Mat& m, types::DType dtype) {
    // produce host tensor with shape {H,W,C}
    int H = m.rows, W = m.cols, C = m.channels();
    std::vector<int> shape = {H, W, C};
    auto t = types::Tensor::AllocateHost(dtype, shape);
    std::size_t copy_bytes = std::min<std::size_t>(m.total() * m.elemSize(), t->byte_size);
    std::memcpy(t->data, m.data, copy_bytes);
    return t;
}

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
    if (!TensorToCvMat(in, in_mat, err)) return false;
    if (in_mat.empty()) {
        out.reset();
        return true;
    }
    cv::Mat resized;
    cv::resize(in_mat, resized, cv::Size(target_w_, target_h_));
    types::DType dt = (in->dtype == types::DType::FLOAT32) ? types::DType::FLOAT32 : types::DType::UINT8;
    out = CvMatToTensor(resized, dt);
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