include(../../../../cmake/find_rknn.cmake)  

# List of RKNN platform sources (engine + runner)
set(RKNN_SRCS
  RknnEngine.cpp
  RknnRunner.cpp
)

# Add RKNN sources directly into the main infer_engine target (scheme A)
# assume infer_engine target already created in parent CMakeLists
target_sources(infer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/RknnEngine.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RknnRunner.cpp
)

# Ensure headers in this folder are visible to infer_engine
target_include_directories(infer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(infer_engine PRIVATE ${LIBRKNNRT})

target_include_directories(infer_engine PRIVATE ${LIBRKNNRT_INCLUDES})
