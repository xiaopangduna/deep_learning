set(UTILS_SRCS
  Log.cpp               
)

add_library(utils STATIC ${UTILS_SRCS})

target_include_directories(utils
  PUBLIC ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# link common/public deps
target_link_libraries(utils
  PUBLIC common
)

# Ensure C++17 for this target
target_compile_features(utils PUBLIC cxx_std_17)

target_link_libraries(utils PUBLIC deploy_global)
