#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>
#include "config/PerceptionConfig.hpp" // 假设该头文件定义了 PerceptionConfig
#include "type/PerceptionData.hpp" // 假设该头文件定义了 PerceptionData 和 PerceptionResult

class PerceptionModel {
protected:
    const PerceptionConfig* config;  // 修改为指针类型
public:
    // 修改构造函数，取引用的地址赋值给指针
    PerceptionModel(const PerceptionConfig& config) : config(&config) {};

    // 定义纯虚函数，用于运行模型，需要在派生类中实现
    virtual void run(const PerceptionData &src, PerceptionResult& dst) = 0;

    // 虚析构函数，确保正确释放派生类对象
    virtual ~PerceptionModel() = default;
};

#endif // MODEL_HPP
