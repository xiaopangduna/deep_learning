#include "processing/Pipeline.hpp"
#include "processing/Transform.hpp"
#include "processing/ImagePipeline.hpp"
#include "common/Tensor.hpp"
#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <vector>
#include <stdexcept> // <-- 新增

using namespace deploy::processing;
using namespace deploy::common;

// Helper: create a TensorBuffer with float32 data from vector<float>
static TensorBuffer make_float_tensor(const std::vector<int>& shape, const std::vector<float>& values) {
    size_t elems = 1;
    for (int d : shape) elems *= d;
    if (values.size() != elems) {
        throw std::runtime_error("make_float_tensor: values.size() does not match shape elements");
    }

    HostAllocator alloc;
    size_t bytes = elems * sizeof(float);
    void* raw = alloc.allocate(bytes);
    if (!raw) {
        throw std::runtime_error("make_float_tensor: allocation failed");
    }
    std::memcpy(raw, values.data(), bytes);

    // wrap with shared_ptr and custom deleter that frees via allocator
    auto deleter = [alloc](void* p) mutable { alloc.deallocate(p); };
    std::shared_ptr<void> data_ptr(raw, deleter);

    TensorBuffer buf;
    buf.data = data_ptr;
    buf.size_in_bytes = bytes;
    buf.desc.shape = shape;
    buf.desc.dtype = DType::FLOAT32;
    buf.desc.layout = Layout::NCHW;
    return buf;
}

// Helper: read float data from TensorBuffer
static std::vector<float> read_float_tensor(const TensorBuffer& buf) {
    size_t elems = 1;
    for (int d : buf.desc.shape) elems *= d;
    const float* fptr = reinterpret_cast<const float*>(buf.data.get());
    return std::vector<float>(fptr, fptr + elems);
}

// A simple Transform implementation that multiplies all float elements by a factor.
class MultiplyTransform : public Transform {
public:
    explicit MultiplyTransform(float factor) : factor_(factor) {}
    bool apply(const Buffer& in, Buffer& out) override {
        if (in.desc.dtype != DType::FLOAT32) return false;
        // allocate output buffer with same shape/dtype/layout
        HostAllocator alloc;
        size_t elems = 1;
        for (int d : in.desc.shape) elems *= d;
        size_t bytes = elems * sizeof(float);
        void* raw = alloc.allocate(bytes);
        if (!raw) return false;
        auto deleter = [alloc](void* p) mutable { alloc.deallocate(p); };
        out.data = std::shared_ptr<void>(raw, deleter);
        out.size_in_bytes = bytes;
        out.desc = in.desc;

        const float* inptr = reinterpret_cast<const float*>(in.data.get());
        float* outptr = reinterpret_cast<float*>(out.data.get());
        for (size_t i = 0; i < elems; ++i) outptr[i] = inptr[i] * factor_;
        return true;
    }
    std::string name() const override { return "Multiply"; }
private:
    float factor_;
}

;

TEST(PipelineTest, ChainMultiplyTransforms) {
    // create input tensor 1x1x2x2 with values [1,2,3,4]
    std::vector<int> shape = {1, 1, 2, 2};
    std::vector<float> vals = {1.f, 2.f, 3.f, 4.f};
    TensorBuffer in = make_float_tensor(shape, vals);

    // pipeline: *2 then *3 => net *6
    ImagePipeline pipe;
    pipe.add_transform(std::make_unique<MultiplyTransform>(2.0f));
    pipe.add_transform(std::make_unique<MultiplyTransform>(3.0f));

    TensorBuffer out;
    ASSERT_TRUE(pipe.run(in, out));

    auto out_vals = read_float_tensor(out);
    ASSERT_EQ(out_vals.size(), vals.size());
    for (size_t i = 0; i < out_vals.size(); ++i) {
        EXPECT_FLOAT_EQ(out_vals[i], vals[i] * 6.0f);
    }
}

TEST(PipelineTest, InPlaceSafeCheck) {
    // Ensure pipeline works when a single transform is applied
    std::vector<int> shape = {1, 1, 2, 2};
    std::vector<float> vals = {0.5f, -1.0f, 2.0f, 4.0f};
    TensorBuffer in = make_float_tensor(shape, vals);

    ImagePipeline pipe;
    pipe.add_transform(std::make_unique<MultiplyTransform>(-1.0f));

    TensorBuffer out;
    ASSERT_TRUE(pipe.run(in, out));
    auto out_vals = read_float_tensor(out);
    for (size_t i = 0; i < out_vals.size(); ++i) {
        EXPECT_FLOAT_EQ(out_vals[i], vals[i] * -1.0f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}