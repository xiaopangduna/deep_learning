#ifndef DEPLOY_UTILS_LOG_HPP_
#define DEPLOY_UTILS_LOG_HPP_

#include <string>
#include <spdlog/spdlog.h>  // 内部依赖 spdlog，对外隐藏

namespace deploy {
namespace utils {

// 日志初始化：在程序入口调用一次（如 main 函数）
// 参数说明：
//   log_level: 日志级别字符串（"debug"|"info"|"warn"|"error"）
//   enable_color: 是否启用控制台彩色输出
void init_logger(const std::string& log_level = "info", bool enable_color = true);

}  // namespace utils
}  // namespace deploy

// 日志宏定义（对外接口，与 spdlog 风格一致但隔离实现）
#define DEPLOY_LOG_DEBUG(...) ::spdlog::debug(__VA_ARGS__)
#define DEPLOY_LOG_INFO(...)  ::spdlog::info(__VA_ARGS__)
#define DEPLOY_LOG_WARN(...)  ::spdlog::warn(__VA_ARGS__)
#define DEPLOY_LOG_ERROR(...) ::spdlog::error(__VA_ARGS__)

// 简化别名（可选，根据团队习惯）
#define LOG_DEBUG DEPLOY_LOG_DEBUG
#define LOG_INFO  DEPLOY_LOG_INFO
#define LOG_WARN  DEPLOY_LOG_WARN
#define LOG_ERROR DEPLOY_LOG_ERROR

#endif  // DEPLOY_UTILS_LOG_HPP_