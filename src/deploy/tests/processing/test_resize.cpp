#include "processing/transforms/Resize.hpp"
#include "common/Tensor.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cstring>

using namespace deploy::processing::transforms;
using namespace deploy::common;

// helper to create NCHW float tensor {1,1,h,w}
static TensorBuffer make_nchw_float(int h, int w, const std::vector<float>& vals) {
    TensorBuffer buf;
    HostAllocator alloc;
    int n = 1, c = 1;
    size_t elems = static_cast<size_t>(n) * c * h * w;
    if (vals.size() != elems) throw std::runtime_error("vals size mismatch");
    size_t bytes = elems * sizeof(float);
    void* raw = alloc.allocate(bytes);
    if (!raw) throw std::runtime_error("alloc failed");
    std::memcpy(raw, vals.data(), bytes);
    auto deleter = [alloc](void* p) mutable { alloc.deallocate(p); };
    buf.data = std::shared_ptr<void>(raw, deleter);
    buf.size_in_bytes = bytes;
    buf.desc.shape = {n, c, h, w};
    buf.desc.dtype = DType::FLOAT32;
    buf.desc.layout = Layout::NCHW;
    return buf;
}

static std::vector<float> read_float(const TensorBuffer& b) {
    size_t elems = 1;
    for (int d : b.desc.shape) elems *= d;
    const float* p = reinterpret_cast<const float*>(b.data.get());
    return std::vector<float>(p, p + elems);
}

TEST(ResizeTest, NearestUpsample2x) {
    // input 2x2: [1,2;3,4] in NCHW
    std::vector<float> invals = {1.f,2.f,3.f,4.f};
    TensorBuffer in = make_nchw_float(2,2,invals);

    ResizeTransform resize(4,4,"nearest");
    TensorBuffer out;
    ASSERT_TRUE(resize.apply(in, out));
    EXPECT_EQ(out.desc.shape.size(), 4);
    EXPECT_EQ(out.desc.shape[2], 4);
    EXPECT_EQ(out.desc.shape[3], 4);
    auto outvals = read_float(out);
    // expected nearest 2x2->4x4 replicate quadrants
    std::vector<float> expect = {
        1,1,2,2,
        1,1,2,2,
        3,3,4,4,
        3,3,4,4
    };
    ASSERT_EQ(outvals.size(), expect.size());
    for (size_t i = 0; i < expect.size(); ++i) {
        EXPECT_FLOAT_EQ(outvals[i], expect[i]);
    }
}