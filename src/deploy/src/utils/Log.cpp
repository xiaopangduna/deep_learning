#include "utils/Log.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <unordered_map>

namespace deploy {
namespace utils {

// 日志级别映射（字符串 -> spdlog 级别）
static const std::unordered_map<std::string, spdlog::level::level_enum> kLogLevelMap = {
    {"debug", spdlog::level::debug},
    {"info",  spdlog::level::info},
    {"warn",  spdlog::level::warn},
    {"error", spdlog::level::error}
};

void init_logger(const std::string& log_level, bool enable_color) {
    // 1. 解析日志级别（默认 info）
    auto level_it = kLogLevelMap.find(log_level);
    spdlog::level::level_enum log_level_enum = spdlog::level::info;
    if (level_it != kLogLevelMap.end()) {
        log_level_enum = level_it->second;
    } else {
        // 日志级别无效时，输出警告并使用默认级别
        ::spdlog::warn("Invalid log level '{}', using default 'info'", log_level);
    }

    // 2. 配置输出目标（控制台，可扩展为文件）
    std::shared_ptr<spdlog::sinks::sink> sink;
    if (enable_color) {
        // 彩色控制台输出（带级别颜色区分）
        sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");  // 带毫秒时间戳
    } else {
        // 无颜色控制台输出（适合日志文件）
        sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
        sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
    }

    // 3. 创建全局日志器
    auto logger = std::make_shared<spdlog::logger>("deploy_logger", sink);
    spdlog::set_default_logger(logger);

    // 4. 设置日志级别和其他属性
    spdlog::set_level(log_level_enum);
    spdlog::flush_on(spdlog::level::error);  // 错误日志立即刷新
}

}  // namespace utils
}  // namespace deploy