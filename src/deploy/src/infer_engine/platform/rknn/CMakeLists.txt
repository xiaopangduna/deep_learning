cmake_minimum_required(VERSION 3.5)

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

# Link RKNN SDK if provided (user can pass RKNN_LIBRARIES/RKNN_INCLUDE_DIR)
if(DEFINED RKNN_INCLUDE_DIR)
  target_include_directories(infer_engine PRIVATE ${RKNN_INCLUDE_DIR})
endif()

if(DEFINED RKNN_LIBRARIES)
  target_link_libraries(infer_engine PRIVATE ${RKNN_LIBRARIES})
else()
  # try to find librknn as fallback
  find_library(_RKNN_LIB NAMES rknn PATHS /usr/lib /usr/local/lib NO_DEFAULT_PATH)
  if(_RKNN_LIB)
    target_link_libraries(infer_engine PRIVATE ${_RKNN_LIB})
  endif()
endif()

# propagate compile definition so platform sources can be conditionally compiled
target_compile_definitions(infer_engine PRIVATE ENABLE_RKNN=1)