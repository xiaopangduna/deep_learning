#ifndef PERCEPTION_DATA_HPP
#define PERCEPTION_DATA_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sstream>
struct PerceptionData
{
    virtual ~PerceptionData() = default;

    // 可选：类型信息，便于工厂识别
    virtual std::string type() const = 0;

    // 可选：通用接口，比如时间戳、数据描述等
    virtual std::string info() const = 0;
};

struct PerceptionResult
{
    virtual ~PerceptionResult() = default;

    // 可选：类型信息，便于工厂识别
    virtual std::string type() const = 0;

    // 可选：通用接口，比如时间戳、数据描述等
    virtual std::string info() const = 0;
};

struct ImageInputData : public PerceptionData
{
    struct ImageItem
    {
        std::string path;
        cv::Mat image;
        int64_t timestamp = 0;
        int64_t frame_id = 0;
        int64_t camera_id = 0;
        int64_t camera_type = 0;
    };

    std::vector<ImageItem> images;

    std::string type() const override
    {
        return "ImageInputData";
    }

    std::string info() const override
    {
        std::ostringstream oss;
        oss << "[ImageInputData] image count = " << images.size();
        return oss.str();
    }
};

struct Yolov5SegmentationResult : public PerceptionResult
{
    struct Item
    {
        std::string path;
        cv::Mat image;
        int64_t timestamp = 0;
    };

    std::vector<Item> images;

    std::string type() const override
    {
        return "Yolov5SegmentationResult";
    }

    std::string info() const override
    {
        std::ostringstream oss;
        oss << "[Yolov5SegmentationResult] image count = " << images.size();
        return oss.str();
    }
};


#endif // PERCEPTION_DATA_HPP
