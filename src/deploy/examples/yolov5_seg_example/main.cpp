// Illustrative yolov5-seg demo (spec-style). Libraries are referenced by expected API.

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <optional>

#include <opencv2/opencv.hpp>

// 直接使用具体模型实现，配置已合并到模型头中
#include "perception_model/segmentation/Yolov5SegModel.hpp"

int main(int argc, char** argv) {
    // 使用命名空间别名，调用更简洁
    namespace dpy = deploy::perception::yolov5_seg;

    dpy::Config cfg;
    cfg.pre.steps = {
        {"Resize",      {{"width","640"}, {"height","640"}, {"mode","letterbox"}, {"device","cuda:0"}}},
        // {"ColorConvert",{{"from","bgr"}, {"to","rgb"}, {"device","cuda:0"}}},
        // {"Normalize",    {{ "mean","0.485,0.456,0.406"}, {"std","0.229,0.224,0.225"}, {"scale","1/255"}, {"device","cpu"}}},
        // {"Permute",      {{"order","HWC->CHW"}, {"device","cpu"}}}
    };
    cfg.engine = {"onnxruntime", {
        {"model_path", "models/yolov5_seg.onnx"},
        {"precision", "fp32"},
        {"device", "cuda:0"},
        {"batch_size", "1"},
        {"dynamic_shape", "false"},
        {"output_names", "boxes,scores,masks"},
        {"optimization_level", "1"} // backend-specific
    }};
    cfg.post.steps = {
        {"DecodeYolo", {{"score_thresh","0.25"}, {"device","cuda:0"}}},
        // {"NMS",        {{"iou_thresh","0.45"}, {"device","cpu"}}}
    };


    // Create（在 Create 中完成初始化），返回 optional<shared_ptr<Yolov5SegModel>>
    auto model_opt = dpy::Yolov5SegModel::Create(cfg);
    if (!model_opt) { std::cerr<<"Create failed\n"; return -1; }
    auto model = std::move(*model_opt);

    // // 3) gather inputs (support multiple image paths)
    // std::vector<cv::Mat> images;
    // if (argc > 1) {
    //     for (int i = 1; i < argc; ++i) {
    //         cv::Mat im = cv::imread(argv[i], cv::IMREAD_COLOR);
    //         if (im.empty()) {
    //             std::cerr << "warning: failed to read image: " << argv[i] << "\n";
    //             continue;
    //         }
    //         images.push_back(std::move(im));
    //     }
    // } else {
    //     const std::string image_path = "examples/yolov5_seg_example/test_image.jpg";
    //     cv::Mat im = cv::imread(image_path, cv::IMREAD_COLOR);
    //     if (im.empty()) {
    //         std::cerr << "read image failed: " << image_path << "\n";
    //         return -1;
    //     }
    //     images.push_back(std::move(im));
    // }

    // if (images.empty()) {
    //     std::cerr << "no valid input images\n";
    //     return -1;
    // }

    // // 4) Predict -> 填充调用方提供的结果结构（约定：bool Predict(const std::vector<cv::Mat>&, std::vector<dpy::Result>&)）
    // std::vector<dpy::Result> results;
    // if (!model->Predict(images, results)) {
    //     std::cerr << "predict failed\n";
    //     return -1;
    // }

    // if (results.size() != images.size()) {
    //     std::cerr << "warning: results count (" << results.size()
    //               << ") != images count (" << images.size() << ")\n";
    // }


    // // 不再显式调用 Shutdown，资源在 model 析构时释放
    return 0;
}