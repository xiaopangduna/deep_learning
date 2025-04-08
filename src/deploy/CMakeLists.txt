# 设置 CMake 最低版本要求，确保使用的 CMake 功能可用
cmake_minimum_required(VERSION 3.10)

# 设置项目名称，这里可以根据实际项目命名
project(DeepLearningDeploy)

# 设置 C++ 标准，这里使用 C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含头文件目录，方便项目中各个源文件找到头文件
include_directories(
    ${PROJECT_SOURCE_DIR}/lib/utils_lib/include/configs
    # 如果有其他头文件目录，可继续添加
)


# 添加库
# 由于是头文件库，无需指定源文件
add_library(utils_lib INTERFACE)

# 设置头文件库的包含目录
target_include_directories(utils_lib INTERFACE
    ${PROJECT_SOURCE_DIR}/lib/utils_lib/include
)

# 添加可执行文件

# 查找 YAML 库
find_package(yaml-cpp REQUIRED)

# 添加可执行文件
add_executable(DeepLearningDeployApp src/app/main.cpp)


# 将 utils_lib 库链接到可执行文件
target_link_libraries(DeepLearningDeployApp PRIVATE utils_lib yaml-cpp)

# 添加测试子目录，让 CMake 处理测试目录下的构建配置
add_subdirectory(tests/utils_lib)

# 如果后续要测试 perception_lib，也可以添加对应的子目录
# add_subdirectory(tests/perception_lib)

set(CMAKE_BUILD_TYPE Debug)