# examples/yolov5_seg_demo/CMakeLists.txt
# 编译YOLOv5分割demo

# 1. 定义可执行目标（目标名建议与目录名一致，避免冲突）
add_executable(yolov5_seg_demo main.cpp)

# 2. 链接依赖（核心库+全局依赖）
target_link_libraries(yolov5_seg_demo
  PRIVATE
    deploy_core       # 项目核心库
    deploy_global     # 全局依赖（OpenCV等）
)

# 3. 头文件路径（引用框架接口）
target_include_directories(yolov5_seg_demo
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# 4. 可选：复制资源文件到编译输出目录（如测试图片、模型）
# （确保运行demo时能找到资源）
set(RESOURCE_FILES "test_image.jpg")
foreach(res ${RESOURCE_FILES})
  configure_file(
    ${res}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${res}  # 复制到bin目录
    COPYONLY
  )
endforeach()