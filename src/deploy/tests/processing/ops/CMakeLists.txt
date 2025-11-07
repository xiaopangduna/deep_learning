add_executable(test_resize test_resize.cpp)
add_executable(demo_resize demo_resize.cpp)

# Link to processing library and common interface (deploy_global) and gtest
# processing target is expected to be defined in src/processing/CMakeLists.txt
target_link_libraries(test_resize PRIVATE processing deploy_global GTest::gtest_main)
target_link_libraries(demo_resize PRIVATE processing deploy_global)

# ensure headers are visible
target_include_directories(test_resize PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(demo_resize PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME ResizeOpTest COMMAND test_resize)
# do not add demo_resize as ctest test; it's for manual inspection