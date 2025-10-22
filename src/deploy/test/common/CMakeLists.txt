# 查找GTest库
find_package(GTest REQUIRED)

# 测试源文件
set(TEST_FILES test_Platform.cpp)

# 生成测试可执行文件
add_executable(test_platform ${TEST_FILES})

# 链接依赖：项目核心库、GTest库
target_link_libraries(test_platform
    deploy_core
    GTest::gtest
    GTest::gtest_main  # 提供main函数，无需自己实现
)

# 添加测试用例（让CTest识别）
add_test(
    NAME test_platform
    COMMAND test_platform
)

# 传递平台宏定义（根据根目录的开关）
if(RK_PLATFORM)
    target_compile_definitions(test_platform PRIVATE RK_PLATFORM)
endif()
if(NVIDIA_PLATFORM)
    target_compile_definitions(test_platform PRIVATE NVIDIA_PLATFORM)
endif()