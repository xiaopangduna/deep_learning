#include "perception_model/segmentation/Yolov5SegModel.hpp"

#include <iostream>
#include <filesystem>

namespace dpy = deploy::perception::yolov5_seg;

struct dpy::Yolov5SegModel::Impl {
    Config cfg;
    bool initialized = false;
    // placeholders for engine / pipelines
    // e.g. std::unique_ptr<Engine> engine;
    // e.g. PreprocessPipeline pre; PostprocessPipeline post;
};

bool dpy::Config::Validate(std::string* err) const {
    if (engine.name.empty()) {
        if (err) *err = "engine.name is empty";
        return false;
    }
    auto it = engine.params.find("model_path");
    if (it == engine.params.end() || it->second.empty()) {
        if (err) *err = "engine.model_path is empty";
        return false;
    }
    if (pre.input_width <= 0 || pre.input_height <= 0) {
        if (err) *err = "invalid input size";
        return false;
    }
    if (post.score_thresh < 0.f || post.score_thresh > 1.f) {
        if (err) *err = "post.score_thresh out of range";
        return false;
    }
    return true;
}

std::optional<std::shared_ptr<dpy::Yolov5SegModel>> dpy::Yolov5SegModel::Create(const Config& cfg) {
    std::string err;
    // optional: cfg validation can be enabled/disabled by caller
    if (!cfg.Validate(&err)) {
        std::cerr << "Config Validate failed: " << err << "\n";
        return std::nullopt;
    }

    // construct lightweight instance
    std::shared_ptr<Yolov5SegModel> inst;
    try {
        inst = std::make_shared<Yolov5SegModel>(cfg);
    } catch (const std::exception& e) {
        std::cerr << "Create failed: ctor exception: " << e.what() << "\n";
        return std::nullopt;
    }

    // call Init to perform heavy / fallible initialization
    if (!inst->Init()) {
        std::cerr << "Create failed: Init failed\n";
        return std::nullopt;
    }

    std::cerr << "Yolov5SegModel created\n";
    return inst;
}

dpy::Yolov5SegModel::Yolov5SegModel(const Config& cfg) : impl_(new Impl()) {
    impl_->cfg = cfg;
}

dpy::Yolov5SegModel::~Yolov5SegModel() noexcept {
    try { Shutdown(); } catch(...) {}
}

void dpy::Yolov5SegModel::Shutdown() noexcept {
    if (!impl_) return;
    if (impl_->initialized) {
        // TODO: release engine / pipeline resources
        impl_->initialized = false;
        std::cerr << "Yolov5SegModel shutdown\n";
    }
}

// Init skeleton: use impl_->cfg (no extra stored fields); keep logic minimal and non-specific
bool dpy::Yolov5SegModel::Init() {
    const Config& cfg = impl_->cfg;

    // 1) basic engine/model_path check (local variable only)
    auto it = cfg.engine.params.find("model_path");
    if (it == cfg.engine.params.end() || it->second.empty()) {
        std::cerr << "Init failed: engine.model_path not set\n";
        return false;
    }
    const std::string model_path = it->second;

    // // 2) simple file existence check
    // if (!std::filesystem::exists(model_path)) {
    //     std::cerr << "Init failed: model file not found: " << model_path << "\n";
    //     return false;
    // }

    // 3) TODO: create engine using cfg.engine.name and cfg.engine.params
    // 4) TODO: build preprocess pipeline from cfg.pre.steps
    // 5) TODO: build postprocess pipeline from cfg.post.steps
    // 6) TODO: optional warmup runs according to cfg.runtime

    // minimal success mark
    impl_->initialized = true;
    std::cerr << "Init succeeded (stub)\n";
    return true;
}

bool dpy::Yolov5SegModel::Predict(const std::vector<cv::Mat>& images, std::vector<Result>& results) {
    if (!impl_ || !impl_->initialized) {
        std::cerr << "Predict called on uninitialized model\n";
        return false;
    }
    // basic contract: results.size() == images.size() on success
    results.clear();
    results.reserve(images.size());

    // Minimal workable mock pipeline:
    // - for each input, perform a simple resize to cfg.pre.input_width/height
    // - produce empty Result (no detections) so main can run end-to-end
    for (const auto& im : images) {
        cv::Mat in = im;
        // if keep_ratio is true, use letterbox-like resize (simple centered resize here)
        cv::Mat resized;
        int w = impl_->cfg.pre.input_width;
        int h = impl_->cfg.pre.input_height;
        if (w > 0 && h > 0) {
            cv::resize(in, resized, cv::Size(w, h));
        } else {
            resized = in.clone();
        }
        // Here you would convert to tensor / run engine / postprocess
        // We return empty detection list as placeholder
        results.emplace_back(Result{});
    }

    return true;
}