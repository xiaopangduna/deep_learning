#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "processing/ops/BgrToRgb.hpp"
#include "processing/ImageCvUtils.hpp"
#include "type/Tensor.hpp"

using namespace deploy::perception::processing;
namespace types = deploy::perception::types;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: demo_bgr2rgb <input_image> <output_image>\n";
        return 2;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];

    cv::Mat src = cv::imread(in_path, cv::IMREAD_UNCHANGED);
    if (src.empty())
    {
        std::cerr << "Failed to read image: " << in_path << "\n";
        return 1;
    }

    // convert cv::Mat -> Tensor using shared util
    types::DType dt = (src.depth() == CV_8U) ? types::DType::UINT8 : types::DType::FLOAT32;
    auto in_t = deploy::perception::processing::CvMatToTensor(src, dt);
    if (!in_t)
    {
        std::cerr << "CvMatToTensor failed\n";
        return 1;
    }

    // setup BgrToRgbOp
    BgrToRgbOp op;
    std::map<std::string, std::string> params; // no params required
    std::string init_err;
    if (!op.Init(params, &init_err))
    {
        std::cerr << "BgrToRgb init error: " << init_err << "\n";
        return 1;
    }

    types::TensorPtr out_t;
    std::string run_err;
    if (!op.Run(in_t, out_t, &run_err))
    {
        std::cerr << "BgrToRgb run error: " << run_err << "\n";
        return 1;
    }
    if (!out_t)
    {
        std::cerr << "BgrToRgb returned null tensor\n";
        return 1;
    }

    // convert output tensor back to cv::Mat using shared util
    cv::Mat result;
    std::string t2m_err;
    if (!deploy::perception::processing::TensorToCvMat(out_t, result, &t2m_err))
    {
        std::cerr << "TensorToCvMat failed: " << t2m_err << "\n";
        return 1;
    }

    if (!cv::imwrite(out_path, result))
    {
        std::cerr << "Failed to write image: " << out_path << "\n";
        return 1;
    }
    std::cout << "Saved BGR->RGB image to " << out_path << "\n";
    return 0;
}