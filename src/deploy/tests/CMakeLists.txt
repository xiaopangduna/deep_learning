# Top-level tests CMake: find GTest, set include dir and add subdirs
# (root CMake already calls enable_testing()/include(CTest))

# find GTest once for all tests (optional: fallback to submodule)
find_package(GTest REQUIRED)

# common include path for tests
set(TEST_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# add per-module test subdirectories
add_subdirectory(common)
add_subdirectory(processing)
add_subdirectory(utils)