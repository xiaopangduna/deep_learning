#ifndef INFER_ENGINE_DEVICE_HPP
#define INFER_ENGINE_DEVICE_HPP

#include <string>

namespace infer_engine {
enum class DeviceType {
    CPU,
    GPU,
    NPU  // 包含RK3588的NPU、昇腾NPU等
};

class Device {
public:
    Device(DeviceType type, int id);
    DeviceType type() const;
    int id() const;
    std::string name() const;  // 设备名称（如"NPU:0"）

private:
    DeviceType type_;
    int device_id_;  // 设备ID（多设备场景下区分）
};
}  // namespace infer_engine

#endif  // INFER_ENGINE_DEVICE_HPP