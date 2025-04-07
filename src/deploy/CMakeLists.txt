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

# 这里可以添加项目的主要可执行文件或库
# 例如，添加一个名为 main_app 的可执行文件
# add_executable(main_app src/main.cpp)

# 添加测试子目录，让 CMake 处理测试目录下的构建配置
add_subdirectory(tests/utils_lib)

# 如果后续要测试 perception_lib，也可以添加对应的子目录
# add_subdirectory(tests/perception_lib)