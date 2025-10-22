#include "utils/Log.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

namespace deploy {
namespace utils {

void LOG(LogLevel level, const std::string& message) {
    // 时间戳
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::string level_str;

    // 级别字符串
    switch (level) {
        case LogLevel::DEBUG: level_str = "[DEBUG]"; break;
        case LogLevel::INFO:  level_str = "[INFO]";  break;
        case LogLevel::WARN:  level_str = "[WARN]";  break;
        case LogLevel::ERROR: level_str = "[ERROR]"; break;
    }

    // 输出格式：[时间] [级别] 消息
    std::cout << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") 
              << "] " << level_str << " " << message << std::endl;
}

}  // namespace utils
}  // namespace deploy