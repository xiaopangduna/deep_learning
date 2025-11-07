# tests/processing/ops: build ops tests (e.g. test_resize.cpp)
cmake_minimum_required(VERSION 3.15)

add_executable(test_resize ops/test_resize.cpp)

# Link to processing library and common interface (deploy_global) and gtest
# processing target is expected to be defined in src/processing/CMakeLists.txt
target_link_libraries(test_resize PRIVATE processing deploy_global GTest::gtest_main)

# ensure headers are visible
target_include_directories(test_resize PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME ResizeOpTest COMMAND test_resize)