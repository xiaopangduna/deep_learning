// #ifndef PERCEPTION_CONFIG_HPP
// #define PERCEPTION_CONFIG_HPP
// #include <iostream>

// // 基类：表示一个简单的二维图形
// struct PerceptionConfig
// {
//     // 声明预处理、推理、后处理结构体，但不实现具体成员
//     struct Preprocessing;
//     struct Inference;
//     struct Postprocessing;

//     // 嵌套的结构体指针成员
//     Preprocessing *preprocessing;
//     Inference *inference;
//     Postprocessing *postprocessing;

//     virtual ~PerceptionConfig()
//     {
//         delete preprocessing;
//         delete inference;
//         delete postprocessing;
//     }
//     // 从 YAML::Node 加载配置，纯虚函数，由派生类实现
//     virtual void loadFromYamlNode(const YAML::Node &node) = 0;
//     virtual void loadFromYaml(const YAML::Node &node) = 0;
//     virtual void printConfig() const = 0;
//     virtual std::string configToString() const = 0;
// };

// // 派生类：表示一个矩形
// struct Yolov5SegmentationModelConfig : public PerceptionConfig
// {
//     struct Preprocessing
//     {
//         double scale;
//         int cropWidth;
//         int cropHeight;
//     };
//     // 实现推理结构体
//     struct Inference
//     {
//         std::string modelPath;
//         std::string device;
//     };
//     // 实现后处理结构体
//     struct Postprocessing
//     {
//         double confidenceThreshold;
//         double nmsThreshold;
//     };
//     Yolov5SegmentationModelConfig()
//     {
//         preprocessing = new Preprocessing();
//         inference = new Inference();
//         postprocessing = new Postprocessing();
//     }
//     // 从 YAML::Node 加载配置
//     void loadFromYamlNode(const YAML::Node &node) override
//     {
//         if (node["preprocessing"])
//         {
//             auto preNode = node["preprocessing"];
//             if (preNode["scale"])
//             {
//                 preprocessing->scale = preNode["scale"].as<double>();
//             }
//             if (preNode["cropWidth"])
//             {
//                 preprocessing->cropWidth = preNode["cropWidth"].as<int>();
//             }
//             if (preNode["cropHeight"])
//             {
//                 preprocessing->cropHeight = preNode["cropHeight"].as<int>();
//             }
//         }
//         if (node["inference"])
//         {
//             auto infNode = node["inference"];
//             if (infNode["modelPath"])
//             {
//                 inference->modelPath = infNode["modelPath"].as<std::string>();
//             }
//             if (infNode["device"])
//             {
//                 inference->device = infNode["device"].as<std::string>();
//             }
//         }
//         if (node["postprocessing"])
//         {
//             auto postNode = node["postprocessing"];
//             if (postNode["confidenceThreshold"])
//             {
//                 postprocessing->confidenceThreshold = postNode["confidenceThreshold"].as<double>();
//             }
//             if (postNode["nmsThreshold"])
//             {
//                 postprocessing->nmsThreshold = postNode["nmsThreshold"].as<double>();
//             }
//         }
//     }
//     // 打印配置信息
//     void printConfig() const override
//     {
//         std::cout << configToString();
//     }

//     // 打印配置信息到字符串
//     std::string configToString() const override
//     {
//         std::ostringstream oss;
//         oss << "Yolov5SegmentationModelConfig:" << std::endl;
//         oss << "  Preprocessing:" << std::endl;
//         oss << "    Scale: " << preprocessing->scale << std::endl;
//         oss << "    Crop Width: " << preprocessing->cropWidth << std::endl;
//         oss << "    Crop Height: " << preprocessing->cropHeight << std::endl;
//         oss << "  Inference:" << std::endl;
//         oss << "    Model Path: " << inference->modelPath << std::endl;
//         oss << "    Device: " << inference->device << std::endl;
//         oss << "  Postprocessing:" << std::endl;
//         oss << "    Confidence Threshold: " << postprocessing->confidenceThreshold << std::endl;
//         oss << "    NMS Threshold: " << postprocessing->nmsThreshold << std::endl;
//         return oss.str();
//     }
// };

// #endif // PERCEPTION_CONFIG_HPP
#ifndef PERCEPTION_CONFIG_HPP
#define PERCEPTION_CONFIG_HPP
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <string>
#include <sstream>

// 抽象的预处理结构体
struct PreprocessingBase {
    virtual ~PreprocessingBase() = default;
    virtual void loadFromYaml(const YAML::Node& node) = 0;
    virtual std::string toString() const = 0;
};

// 抽象的推理结构体
struct InferenceBase {
    virtual ~InferenceBase() = default;
    virtual void loadFromYaml(const YAML::Node& node) = 0;
    virtual std::string toString() const = 0;
};

// 抽象的后处理结构体
struct PostprocessingBase {
    virtual ~PostprocessingBase() = default;
    virtual void loadFromYaml(const YAML::Node& node) = 0;
    virtual std::string toString() const = 0;
};

// 基类：表示一个通用的配置
struct PerceptionConfig {
    PreprocessingBase* preprocessing;
    InferenceBase* inference;
    PostprocessingBase* postprocessing;

    virtual ~PerceptionConfig() {
        delete preprocessing;
        delete inference;
        delete postprocessing;
    }

    // 从 YAML::Node 加载配置，纯虚函数，由派生类实现
    virtual void loadFromYamlNode(const YAML::Node& node) = 0;
    virtual void loadFromYaml(const YAML::Node& node) = 0;
    virtual void printConfig() const = 0;
    virtual std::string configToString() const = 0;
};

// 派生类：表示一个矩形
struct Yolov5SegmentationModelConfig : public PerceptionConfig {
    double width, height;

    // 实现预处理结构体
    struct Preprocessing : public PreprocessingBase {
        double scale;
        int cropWidth;
        int cropHeight;

        void loadFromYaml(const YAML::Node& node) override {
            if (node["scale"]) {
                scale = node["scale"].as<double>();
            }
            if (node["cropWidth"]) {
                cropWidth = node["cropWidth"].as<int>();
            }
            if (node["cropHeight"]) {
                cropHeight = node["cropHeight"].as<int>();
            }
        }

        std::string toString() const override {
            std::ostringstream oss;
            oss << "    Scale: " << scale << std::endl;
            oss << "    Crop Width: " << cropWidth << std::endl;
            oss << "    Crop Height: " << cropHeight << std::endl;
            return oss.str();
        }
    };

    // 实现推理结构体
    struct Inference : public InferenceBase {
        std::string modelPath;
        std::string device;

        void loadFromYaml(const YAML::Node& node) override {
            if (node["modelPath"]) {
                modelPath = node["modelPath"].as<std::string>();
            }
            if (node["device"]) {
                device = node["device"].as<std::string>();
            }
        }

        std::string toString() const override {
            std::ostringstream oss;
            oss << "    Model Path: " << modelPath << std::endl;
            oss << "    Device: " << device << std::endl;
            return oss.str();
        }
    };

    // 实现后处理结构体
    struct Postprocessing : public PostprocessingBase {
        double confidenceThreshold;
        double nmsThreshold;

        void loadFromYaml(const YAML::Node& node) override {
            if (node["confidenceThreshold"]) {
                confidenceThreshold = node["confidenceThreshold"].as<double>();
            }
            if (node["nmsThreshold"]) {
                nmsThreshold = node["nmsThreshold"].as<double>();
            }
        }

        std::string toString() const override {
            std::ostringstream oss;
            oss << "    Confidence Threshold: " << confidenceThreshold << std::endl;
            oss << "    NMS Threshold: " << nmsThreshold << std::endl;
            return oss.str();
        }
    };

    Yolov5SegmentationModelConfig() {
        preprocessing = new Preprocessing();
        inference = new Inference();
        postprocessing = new Postprocessing();
    }

    // 从 YAML::Node 加载配置
    void loadFromYamlNode(const YAML::Node& node) override {
        if (node["width"]) {
            width = node["width"].as<double>();
        }
        if (node["height"]) {
            height = node["height"].as<double>();
        }
        if (node["preprocessing"]) {
            preprocessing->loadFromYaml(node["preprocessing"]);
        }
        if (node["inference"]) {
            inference->loadFromYaml(node["inference"]);
        }
        if (node["postprocessing"]) {
            postprocessing->loadFromYaml(node["postprocessing"]);
        }
    }

    // 从 YAML 文件加载配置
    void loadFromYaml(const YAML::Node& node) override {
        loadFromYamlNode(node);
    }

    // 打印配置信息
    void printConfig() const override {
        std::cout << configToString();
    }

    // 打印配置信息到字符串
    std::string configToString() const override {
        std::ostringstream oss;
        oss << "Yolov5SegmentationModelConfig:" << std::endl;
        oss << "  Width: " << width << std::endl;
        oss << "  Height: " << height << std::endl;
        oss << "  Preprocessing:" << std::endl;
        oss << preprocessing->toString();
        oss << "  Inference:" << std::endl;
        oss << inference->toString();
        oss << "  Postprocessing:" << std::endl;
        oss << postprocessing->toString();
        return oss.str();
    }
};

#endif // PERCEPTION_CONFIG_HPP