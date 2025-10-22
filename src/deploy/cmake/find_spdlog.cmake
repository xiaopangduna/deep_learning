# ============================================
# find_spdlog.cmake
# 自定义 spdlog 查找模块
# ============================================

# 1. 优先检查项目内 third_party 中的 spdlog
set(SPDLOG_THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party/spdlog/include")
set(SPDLOG_HEADER "spdlog/spdlog.h")  # 核心头文件标识

# 2. 查找逻辑：先项目内，再系统路径
if(EXISTS "${SPDLOG_THIRD_PARTY_DIR}/${SPDLOG_HEADER}")
    # 找到项目内版本
    set(SPDLOG_FOUND TRUE)
    set(SPDLOG_INCLUDE_DIRS "${SPDLOG_THIRD_PARTY_DIR}")
    set(SPDLOG_SOURCE "third_party (project-local)")
else()
    # 查找系统安装的 spdlog
    find_path(SPDLOG_INCLUDE_DIRS
        NAMES ${SPDLOG_HEADER}
        PATHS /usr/include /usr/local/include
        DOC "spdlog header file directory"
    )

    if(SPDLOG_INCLUDE_DIRS AND EXISTS "${SPDLOG_INCLUDE_DIRS}/${SPDLOG_HEADER}")
        set(SPDLOG_FOUND TRUE)
        set(SPDLOG_SOURCE "system-wide installation")
    else()
        set(SPDLOG_FOUND FALSE)
        set(SPDLOG_INCLUDE_DIRS "")
        set(SPDLOG_SOURCE "not found")
    endif()
endif()

# ============================================
# 增强打印信息（与 OpenCV 格式保持一致）
# ============================================
if(SPDLOG_FOUND)
    # 查找成功：分模块打印关键信息
    message(STATUS "======================================")
    message(STATUS "spdlog 查找成功!")
    message(STATUS "  来源: ${SPDLOG_SOURCE}")
    message(STATUS "  头文件目录:")
    foreach(inc ${SPDLOG_INCLUDE_DIRS})
        message(STATUS "    - ${inc}")
    endforeach()
    message(STATUS "  特性: header-only (无需链接库)")
    message(STATUS "======================================")
else()
    # 查找失败：给出具体排查建议
    message(FATAL_ERROR "======================================"
        "\nspdlog 查找失败!\n"
        "可能原因及解决方法：\n"
        "1. 未放入项目 third_party，请执行：\n"
        "   git clone https://github.com/gabime/spdlog.git ${PROJECT_SOURCE_DIR}/third_party/spdlog\n"
        "2. 系统未安装，请通过包管理器安装：\n"
        "   sudo apt install libspdlog-dev (Ubuntu/Debian)\n"
        "3. 安装路径非默认，请通过环境变量指定：\n"
        "   export SPDLOG_DIR=/path/to/your/spdlog/install\n"
        "======================================")
endif()