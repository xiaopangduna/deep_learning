# common: public utility/data-structure library used by other modules
# Add implementation files here (Platform.cpp, Tensor.cpp if present, ...)
set(COMMON_SRCS
  Platform.cpp
  # Tensor.cpp          # <-- add if you have implementation file
)

add_library(common STATIC ${COMMON_SRCS})

# Public headers live in top-level include/, implementation-only headers (if any) in this dir
target_include_directories(common
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# Ensure C++17 for this target
target_compile_features(common PUBLIC cxx_std_17)

target_link_libraries(common PUBLIC deploy_global)