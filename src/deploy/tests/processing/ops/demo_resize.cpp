#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "processing/ops/Resize.hpp"
#include "type/Tensor.hpp"

using namespace deploy::perception::processing;
namespace types = deploy::perception::types;

static types::TensorPtr CvMatToTensorCopy(const cv::Mat& m) {
    types::DType dt = (m.depth() == CV_8U) ? types::DType::UINT8 : types::DType::FLOAT32;
    int H = m.rows, W = m.cols, C = m.channels();
    auto t = types::Tensor::AllocateHost(dt, {H, W, C});
    std::size_t copy_bytes = std::min<std::size_t>(t->byte_size, static_cast<std::size_t>(m.total()*m.elemSize()));
    std::memcpy(t->data, m.data, copy_bytes);
    return t;
}

static cv::Mat TensorToCvMatCopy(const types::Tensor& t) {
    if (t.device != "cpu" || !t.data) return {};
    if (t.shape.size() != 3) return {};
    int H = t.shape[0], W = t.shape[1], C = t.shape[2];
    int cv_type = (t.dtype == types::DType::UINT8) ? CV_8UC(C) : CV_32FC(C);
    cv::Mat m(H, W, cv_type);
    std::memcpy(m.data, t.data, std::min<std::size_t>(m.total()*m.elemSize(), t.byte_size));
    return m;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: demo_resize <input_image> <output_image> [width] [height]\n";
        return 2;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    int target_w = (argc >= 4) ? std::stoi(argv[3]) : 64;
    int target_h = (argc >= 5) ? std::stoi(argv[4]) : 64;

    cv::Mat src = cv::imread(in_path, cv::IMREAD_UNCHANGED);
    if (src.empty()) {
        std::cerr << "Failed to read image: " << in_path << "\n";
        return 1;
    }

    auto in_t = CvMatToTensorCopy(src);

    // setup ResizeOp
    ResizeOp op;
    std::map<std::string, std::string> params;
    params["width"] = std::to_string(target_w);
    params["height"] = std::to_string(target_h);
    std::string init_err;
    if (!op.Init(params, &init_err)) {
        std::cerr << "Resize init error: " << init_err << "\n";
        return 1;
    }

    types::TensorPtr out_t;
    std::string run_err;
    if (!op.Run(in_t, out_t, &run_err)) {
        std::cerr << "Resize run error: " << run_err << "\n";
        return 1;
    }
    if (!out_t) {
        std::cerr << "Resize returned null tensor\n";
        return 1;
    }

    cv::Mat result = TensorToCvMatCopy(*out_t);
    if (result.empty()) {
        std::cerr << "Failed to convert output tensor to cv::Mat\n";
        return 1;
    }
    if (!cv::imwrite(out_path, result)) {
        std::cerr << "Failed to write image: " << out_path << "\n";
        return 1;
    }
    std::cout << "Saved resized image to " << out_path << "\n";
    return 0;
}