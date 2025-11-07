#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "processing/ops/Letterbox.hpp"
#include "processing/ImageCvUtils.hpp"
#include "type/Tensor.hpp"

namespace dpp = deploy::perception::processing;
namespace types = deploy::perception::types;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <target_w> <target_h> [bg_color]\n";
        return 2;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    int target_w = std::stoi(argv[3]);
    int target_h = std::stoi(argv[4]);
    int bg_color = (argc >= 6) ? std::stoi(argv[5]) : 114;

    cv::Mat src = cv::imread(in_path, cv::IMREAD_UNCHANGED);
    if (src.empty())
    {
        std::cerr << "Failed to read image: " << in_path << "\n";
        return 1;
    }

    // convert cv::Mat -> Tensor
    types::DType dt = (src.depth() == CV_8U) ? types::DType::UINT8 : types::DType::FLOAT32;
    auto in_t = dpp::CvMatToTensor(src, dt);
    if (!in_t)
    {
        std::cerr << "CvMatToTensor failed\n";
        return 1;
    }

    // prepare and run LetterboxOp
    dpp::LetterboxOp op;
    std::map<std::string, std::string> params;
    params["width"] = std::to_string(target_w);
    params["height"] = std::to_string(target_h);
    params["bg_color"] = std::to_string(bg_color);
    std::string init_err;
    if (!op.Init(params, &init_err))
    {
        std::cerr << "Letterbox init error: " << init_err << "\n";
        return 1;
    }

    dpp::TensorPtr out_t;
    std::string run_err;
    if (!op.Run(in_t, out_t, &run_err))
    {
        std::cerr << "Letterbox run error: " << run_err << "\n";
        return 1;
    }
    if (!out_t)
    {
        std::cerr << "Letterbox returned null tensor\n";
        return 1;
    }

    // convert output tensor -> cv::Mat and save
    cv::Mat result;
    std::string t2m_err;
    if (!dpp::TensorToCvMat(out_t, result, &t2m_err))
    {
        std::cerr << "TensorToCvMat failed: " << t2m_err << "\n";
        return 1;
    }

    if (!cv::imwrite(out_path, result))
    {
        std::cerr << "Failed to write image: " << out_path << "\n";
        return 1;
    }

    std::cout << "Saved letterboxed image to " << out_path << "\n";
}