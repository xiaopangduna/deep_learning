cmake_minimum_required(VERSION 3.10)
project(UtilsLibTests)

set(CMAKE_CXX_STANDARD 11)

# 查找 Google Test 库
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# 查找 yaml-cpp 库
find_package(yaml-cpp REQUIRED)

# 添加可执行文件
add_executable(test_utils_lib_gtest test_config_gtest.cpp)

# 包含头文件目录
target_include_directories(test_utils_lib_gtest PRIVATE ../../lib/utils_lib/include/configs)

# 链接 Google Test 库、pthread 和 yaml-cpp 库
target_link_libraries(test_utils_lib_gtest ${GTEST_LIBRARIES} pthread ${YAML_CPP_LIBRARIES})