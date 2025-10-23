#include "processing/transforms/Resize.hpp"
#include "common/Tensor.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace deploy {
namespace processing {
namespace transforms {

using namespace deploy::common;

bool ResizeTransform::apply(const Buffer& in, Buffer& out) {
    // Minimal implementation: supports only FLOAT32 + NCHW single-batch tensors.
    if (in.desc.dtype != DType::FLOAT32) return false;
    if (in.desc.layout != Layout::NCHW) return false;
    if (in.desc.shape.size() != 4) return false; // expect {N,C,H,W}
    int n = in.desc.shape[0];
    int c = in.desc.shape[1];
    int h = in.desc.shape[2];
    int w = in.desc.shape[3];
    if (n != 1) return false; // single image for now

    int out_h = target_h_;
    int out_w = target_w_;

    size_t elems = static_cast<size_t>(1) * c * out_h * out_w;
    size_t bytes = elems * sizeof(float);

    HostAllocator alloc;
    void* raw = alloc.allocate(bytes);
    if (!raw) return false;
    auto deleter = [alloc](void* p) mutable { alloc.deallocate(p); };
    out.data = std::shared_ptr<void>(raw, deleter);
    out.size_in_bytes = bytes;
    out.desc.shape = {1, c, out_h, out_w};
    out.desc.dtype = DType::FLOAT32;
    out.desc.layout = Layout::NCHW;

    const float* inptr = reinterpret_cast<const float*>(in.data.get());
    float* outptr = reinterpret_cast<float*>(out.data.get());

    // nearest sampling
    for (int ch = 0; ch < c; ++ch) {
        for (int oy = 0; oy < out_h; ++oy) {
            int src_y = std::min(h - 1, static_cast<int>(std::floor((oy + 0.5f) * h / static_cast<float>(out_h))));
            for (int ox = 0; ox < out_w; ++ox) {
                int src_x = std::min(w - 1, static_cast<int>(std::floor((ox + 0.5f) * w / static_cast<float>(out_w))));
                size_t src_idx = static_cast<size_t>(ch) * (h * w) + static_cast<size_t>(src_y) * w + src_x;
                size_t dst_idx = static_cast<size_t>(ch) * (out_h * out_w) + static_cast<size_t>(oy) * out_w + ox;
                outptr[dst_idx] = inptr[src_idx];
            }
        }
    }
    return true;
}

} // namespace transforms
} // namespace processing
} // namespace deploy