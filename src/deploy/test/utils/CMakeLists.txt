cmake_minimum_required(VERSION 3.15)

# 查找GTest
find_package(GTest REQUIRED)

# 收集测试源码（包含日志模块测试）
file(GLOB TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/test_Log.cpp
)

# 生成测试可执行文件
add_executable(test_log ${TEST_SOURCES})

# 链接依赖
target_link_libraries(deploy_test
    GTest::gtest
    GTest::gtest_main
    deploy_core  # 链接项目核心库（包含Log模块）
)

# 添加到CTest
add_test(NAME test_log COMMAND test_log)