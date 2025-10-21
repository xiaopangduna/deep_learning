#include "perception_model/PerceptionModelFactory.hpp"
#include "config/Yolov5SegModelConfig.hpp"
#include <opencv2/opencv.hpp>

int main() {
    // 1. 加载配置
    Yolov5SegModelConfig config;
    config.model_path = "yolov5_seg.rknn";
    config.backend = "rknn";
    config.device_id = 0;
    config.input_h = 640;
    config.input_w = 640;
    config.conf_thresh = 0.25f;
    config.nms_thresh = 0.45f;

    // 2. 创建模型
    auto model = PerceptionModelFactory::create("yolov5_seg", config);
    if (!model->init(config)) {
        return -1;
    }

    // 3. 推理
    cv::Mat img = cv::imread("test.jpg");
    SegmentationResult result;
    if (model->infer(img, result)) {
        // 4. 可视化（绘制边界框和掩码）
        for (auto& det : result.detections) {
            cv::rectangle(img, det.bbox, cv::Scalar(0,255,0), 2);
            // 掩码绘制（简化）
            img(det.bbox).setTo(cv::Scalar(0,0,255), det.mask);
        }
        cv::imwrite("result.jpg", img);
    }

    return 0;
}