#ifndef LOG_HPP
#define LOG_HPP

#include <string>

namespace deploy {
namespace utils {

// 日志级别
enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

// 日志输出函数（简化实现）
void LOG(LogLevel level, const std::string& message);

// 便捷宏定义
#define LOG_DEBUG(msg) LOG(LogLevel::DEBUG, msg)
#define LOG_INFO(msg)  LOG(LogLevel::INFO, msg)
#define LOG_WARN(msg)  LOG(LogLevel::WARN, msg)
#define LOG_ERROR(msg) LOG(LogLevel::ERROR, msg)

}  // namespace utils
}  // namespace deploy

#endif  // LOG_HPP