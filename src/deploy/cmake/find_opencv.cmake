# ============================================
# find_opencv.cmake
# 自定义 OpenCV 查找模块
# ============================================

# 使用系统 OpenCV
find_package(OpenCV 4.5.5 REQUIRED)

# 导出常用变量
set(OpenCV_FOUND TRUE)
set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS ${OpenCV_LIBS})
set(OpenCV_VERSION ${OpenCV_VERSION})

message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")