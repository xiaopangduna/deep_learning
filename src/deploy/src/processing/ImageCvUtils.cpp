#include "processing/ImageCvUtils.hpp"
#include <cstring>

namespace deploy::perception::processing {

namespace types = deploy::perception::types;

bool TensorToCvMat(const TensorPtr& t, cv::Mat& out, std::string* err) {
    if (!t || !t->data) { if (err) *err = "TensorToCvMat: null tensor or data"; return false; }
    if (t->device != "cpu") { if (err) *err = "TensorToCvMat: only cpu tensors supported"; return false; }

    if (t->shape.size() == 3) {
        int H = t->shape[0], W = t->shape[1], C = t->shape[2];
        int cv_type = (t->dtype == types::DType::UINT8) ? CV_8UC(C) :
                      (t->dtype == types::DType::FLOAT32) ? CV_32FC(C) : -1;
        if (cv_type < 0) { if (err) *err = "TensorToCvMat: unsupported dtype"; return false; }
        out = cv::Mat(H, W, cv_type);
        std::size_t copy_bytes = std::min<std::size_t>(out.total() * out.elemSize(), t->byte_size);
        std::memcpy(out.data, t->data, copy_bytes);
        return true;
    }

    if (t->shape.size() == 4 && t->shape[0] == 1) {
        int C = t->shape[1], H = t->shape[2], W = t->shape[3];
        int cv_type = (t->dtype == types::DType::UINT8) ? CV_8UC(C) :
                      (t->dtype == types::DType::FLOAT32) ? CV_32FC(C) : -1;
        if (cv_type < 0) { if (err) *err = "TensorToCvMat: unsupported dtype"; return false; }
        out = cv::Mat(H, W, cv_type);
        if (t->dtype == types::DType::UINT8) {
            auto src = static_cast<uint8_t*>(t->data);
            for (int c = 0; c < C; ++c) {
                size_t plane = static_cast<size_t>(c) * H * W;
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        out.data[(h * W + w) * C + c] = src[plane + h * W + w];
            }
        } else {
            auto src = static_cast<float*>(t->data);
            auto dst = reinterpret_cast<float*>(out.data);
            for (int c = 0; c < C; ++c) {
                size_t plane = static_cast<size_t>(c) * H * W;
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        dst[(h * W + w) * C + c] = src[plane + h * W + w];
            }
        }
        return true;
    }

    if (err) *err = "TensorToCvMat: unsupported tensor shape";
    return false;
}

TensorPtr CvMatToTensor(const cv::Mat& m, types::DType dtype) {
    int H = m.rows, W = m.cols, C = m.channels();
    std::vector<int> shape = {H, W, C};
    auto t = types::Tensor::AllocateHost(dtype, shape);
    std::size_t copy_bytes = std::min<std::size_t>(m.total() * m.elemSize(), t->byte_size);
    std::memcpy(t->data, m.data, copy_bytes);
    return t;
}

} // namespace deploy::perception::processing