cmake_minimum_required(VERSION 3.15)

# perception_model library: high-level model wrapper that composes processing + infer_engine
add_library(perception_model STATIC
  PerceptionModelFactory.cpp
  segmentation/Yolov5SegModel.cpp
)

target_include_directories(perception_model
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against internal modules
target_link_libraries(perception_model
  PRIVATE
    infer_engine      # depends on infer_engine target in the same project
    processing        # depends on processing target
)

# Link OpenCV if used by the model implementation
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
  target_link_libraries(perception_model PRIVATE ${OpenCV_LIBS})
endif()

# If perception_model exposes a public API that depends on infer_engine/processing headers
# the PUBLIC include above ensures consumers can include the headers from include/.
```# filepath: /home/orangepi/HectorHuang/deep_learning/src/deploy/src/perception_model/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# perception_model library: high-level model wrapper that composes processing + infer_engine
add_library(perception_model STATIC
  PerceptionModelFactory.cpp
  segmentation/Yolov5SegModel.cpp
)

target_include_directories(perception_model
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against internal modules
target_link_libraries(perception_model
  PRIVATE
    infer_engine      # depends on infer_engine target in the same project
    processing        # depends on processing target
)

# Link OpenCV if used by the model implementation
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
  target_link_libraries(perception_model PRIVATE ${OpenCV_LIBS})
endif()

# If perception_model exposes a public API that depends on infer_engine/processing headers
# the PUBLIC include above ensures consumers can include the headers from