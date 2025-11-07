#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "processing/ops/Letterbox.hpp"
#include "processing/ops/BgrToRgb.hpp"
#include "processing/ops/PackToRknn.hpp"
#include "processing/ImageCvUtils.hpp"
#include "type/Tensor.hpp"

namespace dpp = deploy::perception::processing;
namespace types = deploy::perception::types;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: demo_pack_to_rknn <input_image> <out_image> <target_w> <target_h>\n";
        return 2;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    int target_w = std::stoi(argv[3]);
    int target_h = std::stoi(argv[4]);
    cv::Mat src = cv::imread(in_path, cv::IMREAD_UNCHANGED);
    if (src.empty()) {
        std::cerr << "Failed to read image\n";
        return 1;
    }

    // cv::Mat -> Tensor
    types::DType dt = (src.depth() == CV_8U) ? types::DType::UINT8 : types::DType::FLOAT32;
    auto in_t = dpp::CvMatToTensor(src, dt);
    if (!in_t) { std::cerr << "CvMatToTensor failed\n"; return 1; }

    // Letterbox
    dpp::LetterboxOp lbop;
    std::map<std::string,std::string> lb_params;
    lb_params["width"] = std::to_string(target_w);
    lb_params["height"] = std::to_string(target_h);
    lbop.Init(lb_params, nullptr);
    dpp::TensorPtr t_lb;
    if (!lbop.Run(in_t, t_lb, nullptr)) { std::cerr << "Letterbox failed\n"; return 1; }

    // BGR -> RGB
    dpp::BgrToRgbOp b2rop;
    b2rop.Init({}, nullptr);
    dpp::TensorPtr t_rgb;
    if (!b2rop.Run(t_lb, t_rgb, nullptr)) { std::cerr << "BgrToRgb failed\n"; return 1; }

    // Pack to RKNN format (UINT8 NHWC)
    dpp::PackToRknnOp packop;
    std::map<std::string,std::string> pack_params;
    pack_params["model_fmt"] = "UINT8_NHWC";
    packop.Init(pack_params, nullptr);
    dpp::TensorPtr t_pack;
    if (!packop.Run(t_rgb, t_pack, nullptr)) { std::cerr << "PackToRknn failed\n"; return 1; }

    // Convert packed tensor back to cv::Mat and save for verification
    cv::Mat result;
    std::string err;
    if (!dpp::TensorToCvMat(t_pack, result, &err)) {
        std::cerr << "TensorToCvMat failed: " << err << "\n";
        return 1;
    }
    if (!cv::imwrite(out_path, result)) {
        std::cerr << "Failed to write output\n";
        return 1;
    }
    std::cout << "Saved packed image to " << out_path << "\n";
    return 0;
}