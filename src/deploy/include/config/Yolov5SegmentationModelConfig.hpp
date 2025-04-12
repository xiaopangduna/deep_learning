#pragma once
#include "config/PerceptionConfig.hpp"

// 派生类：表示一个矩形
struct Yolov5SegmentationModelConfig : public PerceptionConfig
{

    // 实现预处理结构体
    struct Preprocessing
    {
        std::string device;
        int image_size;
        int batch_size;
        void loadFromYamlNode(const YAML::Node &node)
        {
            loadValueFromYaml(node, "device", device);
            loadValueFromYaml(node, "image_size", image_size);
            loadValueFromYaml(node, "batch_size", batch_size);
        }       


        std::string toString() const
        {
            std::ostringstream oss;
            oss << "    device: " << device << std::endl;
            oss << "    image_size: " << image_size << std::endl;
            oss << "    batch_size: " << batch_size << std::endl;
            return oss.str();
        }
    };

    // 实现推理结构体
    struct Inference
    {
        std::string device;
        std::string path_to_model;

        void loadFromYamlNode(const YAML::Node &node)
        {
            loadValueFromYaml(node, "device", device);
            loadValueFromYaml(node, "path_to_model", path_to_model);
        }

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "    Device: " << device << std::endl;
            oss << "    Model Path: " << path_to_model << std::endl;
            return oss.str();
        }
    };

    // 实现后处理结构体
    struct Postprocessing
    {
        double conf_threshold;

        void loadFromYamlNode(const YAML::Node &node)
        {
            loadValueFromYaml(node, "conf_threshold", conf_threshold);
        }

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "    conf_threshold: " << conf_threshold << std::endl;
            return oss.str();
        }
    };
    // 成员变量
    Preprocessing preprocessing;
    Inference inference;
    Postprocessing postprocessing;


    // 从 YAML::Node 加载配置
    void
    loadFromYamlNode(const YAML::Node &node) override
    {

        if (node["preprocessing"])
        {
            preprocessing.loadFromYamlNode(node["preprocessing"]);
        }
        if (node["inference"])
        {
            inference.loadFromYamlNode(node["inference"]);
        }
        if (node["postprocessing"])
        {
            postprocessing.loadFromYamlNode(node["postprocessing"]);
        }
    }


    // 打印配置信息
    void printConfig() const override
    {
        std::cout << configToString();
    }

    // 打印配置信息到字符串
    std::string configToString() const override
    {
        std::ostringstream oss;
        oss << "Yolov5SegmentationModelConfig:" << std::endl;
        oss << "  Preprocessing:" << std::endl;
        oss << preprocessing.toString();
        oss << "  Inference:" << std::endl;
        oss << inference.toString();
        oss << "  Postprocessing:" << std::endl;
        oss << postprocessing.toString();
        return oss.str();
    }
};
