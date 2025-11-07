#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <cstring>

#include "processing/ops/Resize.hpp"
#include "type/Tensor.hpp"

using namespace deploy::perception::processing;
namespace types = deploy::perception::types;

static types::TensorPtr MakeUint8HWCTensor(int H, int W, int C, const cv::Scalar& fill = cv::Scalar(0,0,0)) {
    cv::Mat m(H, W, CV_MAKETYPE(CV_8U, C), fill);
    auto t = types::Tensor::AllocateHost(types::DType::UINT8, {H, W, C});
    std::memcpy(t->data, m.data, std::min(t->byte_size, static_cast<std::size_t>(m.total()*m.elemSize())));
    return t;
}

TEST(ResizeOpTest, InitInvalidParams) {
    ResizeOp op;
    std::map<std::string, std::string> bad;
    bad["width"] = "0";
    bad["height"] = "-1";
    std::string err;
    EXPECT_FALSE(op.Init(bad, &err));
    EXPECT_FALSE(err.empty());
}

TEST(ResizeOpTest, ResizeDownscaleUint8) {
    // input 100x200x3, resize to 25x50
    auto in = MakeUint8HWCTensor(100, 200, 3, cv::Scalar(127, 64, 32));
    ResizeOp op;
    std::map<std::string, std::string> params;
    params["width"] = "50";
    params["height"] = "25";
    std::string init_err;
    ASSERT_TRUE(op.Init(params, &init_err)) << init_err;

    types::TensorPtr out;
    std::string run_err;
    ASSERT_TRUE(op.Run(in, out, &run_err)) << run_err;
    ASSERT_NE(out, nullptr);

    ASSERT_EQ(out->shape.size(), 3u);
    EXPECT_EQ(out->shape[0], 25);
    EXPECT_EQ(out->shape[1], 50);
    EXPECT_EQ(out->shape[2], 3);
    EXPECT_EQ(out->dtype, types::DType::UINT8);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}