set(INFER_ENGINE_SRCS
  Factory.cpp               
)

add_library(infer_engine STATIC ${INFER_ENGINE_SRCS})

target_include_directories(infer_engine
  PUBLIC ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# link common/public deps
target_link_libraries(infer_engine
  PUBLIC common
)

# Ensure C++17 for this target
target_compile_features(infer_engine PUBLIC cxx_std_17)

target_link_libraries(infer_engine PUBLIC deploy_global)

add_subdirectory(platform)