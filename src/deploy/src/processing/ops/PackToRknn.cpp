#include "processing/ops/PackToRknn.hpp"
#include "type/Tensor.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace deploy::perception::processing {

namespace types = deploy::perception::types;

bool PackToRknnOp::Init(const Params& params, std::string* err) {
    if (params.count("model_fmt")) model_fmt_ = params.at("model_fmt");
    // Only support UINT8_NHWC for now
    if (model_fmt_ != "UINT8_NHWC") {
        if (err) *err = "PackToRknnOp: only model_fmt=UINT8_NHWC supported";
        return false;
    }
    return true;
}

static bool is_hwc(const types::TensorPtr& t) {
    return (t && t->shape.size() == 3);
}
static bool is_chw_batch1(const types::TensorPtr& t) {
    return (t && t->shape.size() == 4 && t->shape[0] == 1);
}

bool PackToRknnOp::Run(const TensorPtr& in, TensorPtr& out, std::string* err) const {
    if (!in) {
        if (err) *err = "PackToRknnOp: input tensor is null";
        return false;
    }
    // Determine input geometry and channels
    int H = 0, W = 0, C = 0;
    bool input_is_hwc = false;
    if (is_hwc(in)) {
        H = in->shape[0]; W = in->shape[1]; C = in->shape[2];
        input_is_hwc = true;
    } else if (is_chw_batch1(in)) {
        C = in->shape[1]; H = in->shape[2]; W = in->shape[3];
        input_is_hwc = false;
    } else {
        if (err) *err = "PackToRknnOp: unsupported input tensor shape";
        return false;
    }
    if (H <= 0 || W <= 0 || C <= 0) {
        if (err) *err = "PackToRknnOp: invalid hwc";
        return false;
    }

    // If input is already UINT8 HWC and contiguous, reuse it.
    if (in->dtype == types::DType::UINT8 && input_is_hwc) {
        // Return the same tensor (caller must know ownership semantics)
        out = in;
        out->device = "cpu";
        return true;
    }

    // Allocate output host tensor with shape {H,W,C} and UINT8 dtype
    std::vector<int> out_shape = {H, W, C};
    auto t_out = types::Tensor::AllocateHost(types::DType::UINT8, out_shape);
    if (!t_out || !t_out->data) {
        if (err) *err = "PackToRknnOp: AllocateHost failed";
        return false;
    }
    t_out->device = "cpu";

    size_t plane_size = static_cast<size_t>(H) * W;
    uint8_t* dst = static_cast<uint8_t*>(t_out->data);

    if (in->dtype == types::DType::UINT8) {
        // Input is UINT8 but maybe CHW -> perform reorder
        uint8_t* src = static_cast<uint8_t*>(in->data);
        if (input_is_hwc) {
            // Should not reach here because handled above, but safe memcpy
            size_t bytes = std::min(t_out->byte_size, in->byte_size);
            std::memcpy(dst, src, bytes);
        } else {
            // CHW (1,C,H,W) -> HWC
            for (int c = 0; c < C; ++c) {
                size_t plane_off = static_cast<size_t>(c) * plane_size;
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        size_t dst_idx = (static_cast<size_t>(h) * W + w) * C + c;
                        dst[dst_idx] = src[plane_off + static_cast<size_t>(h) * W + w];
                    }
                }
            }
        }
        out = t_out;
        return true;
    } else if (in->dtype == types::DType::FLOAT32) {
        // Convert float -> uint8 (no normalization), support CHW or HWC
        float* srcf = static_cast<float*>(in->data);
        if (input_is_hwc) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        size_t src_idx = (static_cast<size_t>(h) * W + w) * C + c;
                        float v = srcf[src_idx];
                        int iv = static_cast<int>(std::lround(v));
                        iv = std::min(255, std::max(0, iv));
                        dst[src_idx] = static_cast<uint8_t>(iv);
                    }
                }
            }
        } else {
            // CHW -> HWC + cast
            for (int c = 0; c < C; ++c) {
                size_t plane_off = static_cast<size_t>(c) * plane_size;
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        size_t dst_idx = (static_cast<size_t>(h) * W + w) * C + c;
                        float v = srcf[plane_off + static_cast<size_t>(h) * W + w];
                        int iv = static_cast<int>(std::lround(v));
                        iv = std::min(255, std::max(0, iv));
                        dst[dst_idx] = static_cast<uint8_t>(iv);
                    }
                }
            }
        }
        out = t_out;
        return true;
    } else {
        if (err) *err = "PackToRknnOp: unsupported input dtype";
        return false;
    }
}

} // namespace deploy::perception::processing