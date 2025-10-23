# processing tests
add_executable(test_processing test_processing.cpp)
target_include_directories(test_processing PRIVATE ${TEST_INCLUDE_DIR})
target_link_libraries(test_processing PRIVATE processing_core GTest::gtest_main)
add_test(NAME test_processing COMMAND test_processing)

add_executable(test_resize test_resize.cpp)
target_include_directories(test_resize PRIVATE ${TEST_INCLUDE_DIR})
target_link_libraries(test_resize PRIVATE processing_core GTest::gtest_main)
add_test(NAME test_resize COMMAND test_resize)