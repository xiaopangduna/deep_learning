#pragma once
#include <memory>
#include <string>
#include <functional>
#include "type/Tensor.hpp"
#include "type/Step.hpp"

namespace deploy::perception::processing {

using TensorPtr = std::shared_ptr<deploy::perception::types::Tensor>;
using Params = deploy::perception::types::Params;

// 抽象算子：Init 进行参数初始化，Run 接受一个输入 TensorPtr，输出结果 TensorPtr。
// 不依赖全局错误字符串，返回 bool 表示成功。
class Op {
public:
    virtual ~Op() = default;
    virtual bool Init(const Params& params, std::string* err = nullptr) = 0;
    virtual bool Run(const TensorPtr& in, TensorPtr& out, std::string* err = nullptr) const = 0;
    virtual std::string name() const noexcept = 0;
};
using OpPtr = std::unique_ptr<Op>;

// 工厂函数类型：根据参数创建已初始化的 Op（失败返回 nullptr 并可写 err）
using OpCreateFn = std::function<OpPtr(const Params& params, std::string* err)>;

// 注册/查找接口（实现放在 OpRegistry.cpp）
// - RegisterOpFactory: 在进程启动/模块加载时注册某种 Op 的创建函数（线程安全）
// - CreateOpByName: 根据名字创建并返回 OpPtr（或 nullptr）
bool RegisterOpFactory(const std::string& op_name, OpCreateFn fn);
OpPtr CreateOpByName(const std::string& op_name, const Params& params, std::string* err = nullptr);

} // namespace deploy::perception::processing