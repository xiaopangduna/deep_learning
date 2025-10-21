#include "infer_engine/Device.hpp"

namespace infer_engine {
Device::Device(DeviceType type, int id) : type_(type), device_id_(id) {}

DeviceType Device::type() const {
    return type_;
}

int Device::id() const {
    return device_id_;
}

std::string Device::name() const {
    // 生成设备名称（如"NPU:0"、"CPU:0"）
    std::string type_str;
    switch (type_) {
        case DeviceType::CPU: type_str = "CPU"; break;
        case DeviceType::GPU: type_str = "GPU"; break;
        case DeviceType::NPU: type_str = "NPU"; break;
        default: type_str = "UNKNOWN";
    }
    return type_str + ":" + std::to_string(device_id_);
}
}  // namespace infer_engine