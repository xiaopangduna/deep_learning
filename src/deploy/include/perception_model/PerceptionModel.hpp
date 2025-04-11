// model.hpp
#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>
#include "config/PerceptionConfig.hpp" // 假设该头文件定义了 PerceptionConfig

class PerceptionModel {
protected:
    const PerceptionConfig* config;  // 修改为指针类型
public:
    PerceptionModel(const PerceptionConfig* config) : config(config) {}

    // // 定义纯虚函数，用于加载模型，需要在派生类中实现
    // virtual void loadModel(const std::string& modelPath) = 0;

    // // 定义纯虚函数，用于进行推理，需要在派生类中实现
    // virtual void infer() = 0;

    // 定义纯虚函数，用于运行模型，需要在派生类中实现
    virtual void run() = 0;

    // 虚析构函数，确保正确释放派生类对象
    virtual ~PerceptionModel() = default;

// protected:
//     // 保护成员变量，存储感知配置
//     PerceptionConfig config;
};

#endif // MODEL_HPP
