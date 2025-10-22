# 1. 基础配置：指定最低CMake版本和项目元信息
#    最低版本选3.15（支持现代target特性和接口传递）
cmake_minimum_required(VERSION 3.15)
project(deploy_framework 
  VERSION 0.1.0 
  LANGUAGES CXX
  DESCRIPTION "Deep learning deployment framework for embedded devices (support RKNN/OpenCV)"
)


# 2. 全局编译环境配置（跨目录生效）
#    统一C++标准（必须17，支持现代语法如std::optional、结构化绑定）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用编译器扩展（如gnu++17）

# 统一输出路径（所有子目录的二进制文件集中存放）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)  # 可执行文件
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)  # 动态库
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)  # 静态库


# 3. 项目级功能开关（控制子模块编译，默认只开启核心功能）
#    开关命名规则：<动作>_<模块>，如BUILD_EXAMPLES、ENABLE_RKNN
option(BUILD_EXAMPLES "Build example programs (verify framework)" ON)
option(ENABLE_RKNN "Enable RKNN backend support (for Rockchip NPU)" OFF)  # 初期默认关闭
option(ENABLE_DEBUG "Enable debug mode (with -g and verbose log)" OFF)


# （2）环境检测变量（供子目录判断平台/版本）
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(PLATFORM_ARM64 TRUE)  # 子目录可根据此变量适配ARM代码
endif()

# 3. 基础依赖管理（传递给所有子目录）
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(cmake/find_opencv.cmake)  # 自定义 OpenCV 查找模块
include(cmake/find_yaml-cpp.cmake)  # 自定义 YAML-CPP 查找模块
include(cmake/find_spdlog.cmake)  # 导入 spdlog 查找逻辑

# 用INTERFACE目标统一传递依赖（子目录只需链接此目标）
add_library(deploy_global INTERFACE)
# 链接库
target_link_libraries(deploy_global INTERFACE
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBS}           # 加入 yaml-cpp 库
)
# 包含头文件
target_include_directories(deploy_global INTERFACE
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}   # 加入 yaml-cpp 头文件路径
    ${SPDLOG_INCLUDE_DIRS}     # 加入 spdlog 头文件路径
)

# 4. 子目录管理（只决定“编译哪些目录”，不干涉目录内逻辑）
add_subdirectory(src)  # 核心库（必选）

if(BUILD_EXAMPLES)
  add_subdirectory(examples)  # 示例程序（可选，由开关控制）
endif()

# 编译测试代码（test目录下的代码）
enable_testing()  # 启用测试
add_subdirectory(test)