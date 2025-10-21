# ============================================
# find_yaml-cpp.cmake
# 自定义 yaml-cpp 查找模块
# ============================================

# 尝试使用 pkg-config 查找
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(YAML_CPP yaml-cpp)
    if(YAML_CPP_FOUND)
        set(YAML_CPP_INCLUDE_DIRS ${YAML_CPP_INCLUDE_DIRS})
        set(YAML_CPP_LIBS ${YAML_CPP_LIBRARIES})
    endif()
endif()

# 查找系统默认路径
if(NOT YAML_CPP_FOUND)
    find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h HINTS /usr/include)
    find_library(YAML_CPP_LIBRARY NAMES yaml-cpp PATHS /usr/lib/x86_64-linux-gnu)
    if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
        set(YAML_CPP_FOUND TRUE)
        set(YAML_CPP_INCLUDE_DIRS ${YAML_CPP_INCLUDE_DIR})
        set(YAML_CPP_LIBS ${YAML_CPP_LIBRARY})
    endif()
endif()

# ============================================
# 增强打印信息
# ============================================
if(YAML_CPP_FOUND)
    message(STATUS "======================================")
    message(STATUS "yaml-cpp 查找成功!")
    message(STATUS "  头文件目录: ${YAML_CPP_INCLUDE_DIRS}")
    message(STATUS "  链接库: ${YAML_CPP_LIBS}")
    message(STATUS "======================================")
else()
    message(FATAL_ERROR "======================================"
        "\nyaml-cpp 查找失败!\n"
        "可能原因及解决方法：\n"
        "1. 未安装 yaml-cpp，请先通过系统包或源码安装\n"
        "2. 如果安装在非标准目录，请通过环境变量指定：\n"
        "   -DYAML_CPP_INCLUDE_DIR=/path/to/include \\\n"
        "   -DYAML_CPP_LIBRARY=/path/to/libyaml-cpp.so\n"
        "3. 检查 pkg-config 是否可用，确保 `yaml-cpp.pc` 在 PKG_CONFIG_PATH 中\n"
        "======================================")
endif()
