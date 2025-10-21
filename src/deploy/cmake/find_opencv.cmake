# ============================================
# find_opencv.cmake
# 自定义 OpenCV 查找模块
# ============================================

# 使用系统 OpenCV
find_package(OpenCV REQUIRED)

# 导出常用变量
set(OpenCV_FOUND TRUE)
set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS ${OpenCV_LIBS})
set(OpenCV_VERSION ${OpenCV_VERSION})
set(OpenCV_COMPONENTS ${OpenCV_COMPONENTS})  # 新增：导出找到的组件列表

# ============================================
# 增强打印信息（核心优化部分）
# ============================================
if(OpenCV_FOUND)
  # 查找成功：分模块打印关键信息
  message(STATUS "======================================")
  message(STATUS "OpenCV 查找成功!")
  message(STATUS "  版本: ${OpenCV_VERSION}")
  message(STATUS "  头文件目录:")
  foreach(inc ${OpenCV_INCLUDE_DIRS})
    message(STATUS "    - ${inc}")
  endforeach()
  message(STATUS "  找到的组件: ${OpenCV_COMPONENTS}")
  message(STATUS "  链接库列表:")
  foreach(lib ${OpenCV_LIBS})
    message(STATUS "    - ${lib}")
  endforeach()
  message(STATUS "======================================")
else()
  # 查找失败：给出更具体的排查建议
  message(FATAL_ERROR "======================================"
    "\nOpenCV 查找失败!\n"
    "可能原因及解决方法：\n"
    "1. 未安装 OpenCV，请先安装对应版本\n"
    "2. 安装路径非默认，请通过环境变量指定：\n"
    "   export OpenCV_DIR=/path/to/your/opencv/install\n"
    "3. 缺少必需组件，请重新安装 OpenCV 并包含核心组件（core, imgproc 等）\n"
    "======================================")
endif()