# 设置 CMake 最低版本要求，确保使用的 CMake 功能可用
cmake_minimum_required(VERSION 3.10)

# 设置项目名称，这里可以根据实际项目命名
project(DeepLearningDeploy)

# 设置 C++ 标准，这里使用 C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义源文件列表
set(SOURCES
    src/app/main.cpp
    src/perception_model/PerceptionModelFactory.cpp
    src/perception_model/Yolov5SegmentationModel.cpp
)

# 添加可执行文件
add_executable(DeepLearningDeployApp ${SOURCES})

# 设置编译选项
if(NOT CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif()

# 设置头文件库
add_library(utils_lib INTERFACE)
target_include_directories(utils_lib INTERFACE
    ${PROJECT_SOURCE_DIR}/lib/utils_lib/include
)

# 查找 YAML 库
find_package(yaml-cpp REQUIRED)

# 设置可执行文件的包含目录
target_include_directories(DeepLearningDeployApp PRIVATE
    ${PROJECT_SOURCE_DIR}/lib/utils_lib/include
    ${PROJECT_SOURCE_DIR}/include
)

# 链接库到可执行文件
target_link_libraries(DeepLearningDeployApp PRIVATE utils_lib yaml-cpp)

# 添加测试子目录，让 CMake 处理测试目录下的构建配置
add_subdirectory(tests/utils_lib)

# 如果后续要测试 perception_lib，也可以添加对应的子目录
# add_subdirectory(tests/perception_lib)

# 设置构建类型为 Debug
set(CMAKE_BUILD_TYPE Debug)

