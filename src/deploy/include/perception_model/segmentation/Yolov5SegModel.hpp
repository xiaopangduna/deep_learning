#pragma once
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <memory>
#include <opencv2/opencv.hpp>

namespace deploy::perception::yolov5_seg {

// key-value params for steps / engine
using Params = std::map<std::string,std::string>;
struct Step { std::string name; Params params; };

// Pre/Post step config holder (keeps flexibility)
struct PreprocessConfig {
    int input_width  = 640;
    int input_height = 640;
    bool keep_ratio  = true;
    std::vector<Step> steps;
};
struct EngineStep { std::string name; Params params; };
struct PostprocessConfig {
    float score_thresh = 0.25f;
    float iou_thresh   = 0.45f;
    std::vector<Step> steps;
};
struct RuntimeOptions { int num_threads = 4; int warmup_runs = 0; };

// Full model config (model-specific)
struct Config {
    PreprocessConfig pre;
    EngineStep engine; // name + params
    PostprocessConfig post;
    RuntimeOptions runtime;

    // simple validation
    bool Validate(std::string* err = nullptr) const;
};

// simple inference result types (public)
struct Detection {
    int x1=0, y1=0, x2=0, y2=0;
    std::string label;
    float score = 0.f;
};
struct Result {
    std::vector<Detection> detections;
    // extensible: masks, meta, timings...
};

// Yolov5SegModel public API
class Yolov5SegModel {
public:
    // construct+init; return nullopt on failure
    static std::optional<std::shared_ptr<Yolov5SegModel>> Create(const Config& cfg);

    // batch predict: fill results (one entry per input image). return false on failure.
    bool Predict(const std::vector<cv::Mat>& images, std::vector<Result>& results);

    // explicit release (also called in dtor)
    void Shutdown() noexcept;

    ~Yolov5SegModel() noexcept;

private:
    explicit Yolov5SegModel(const Config& cfg);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace deploy::perception::yolov5_seg

