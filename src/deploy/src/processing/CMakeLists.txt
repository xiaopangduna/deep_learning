set(PROCESSING_SRCS
  Factory.cpp               
  Op.cpp
  ops/Resize.cpp   
)

add_library(processing STATIC ${PROCESSING_SRCS})

target_include_directories(processing
  PUBLIC ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# link common/public deps
target_link_libraries(processing
  PUBLIC common
)

# Ensure C++17 for this target
target_compile_features(processing PUBLIC cxx_std_17)

target_link_libraries(processing PUBLIC deploy_global)