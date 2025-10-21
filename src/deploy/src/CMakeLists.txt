# src/CMakeLists.txt

# 1. 收集核心源文件（根据你的实际文件路径修改）
set(CORE_SRCS
  core/Device.cpp
  core/ModelInfo.cpp
  perception_model/PerceptionModelFactory.cpp
  perception_model/Yolov5SegModel.cpp
  preprocess/Yolov5SegPreprocess.cpp
)

# 2. 排除未实现的文件（如果某些文件还没写，先注释掉，避免编译错误）
# 例如：如果RknnEngine.cpp还没准备好，暂时不加入
# list(APPEND CORE_SRCS backend/rknn/RknnEngine.cpp)

# 3. 生成核心共享库（必须定义，否则没有编译目标）
add_library(deploy_core SHARED ${CORE_SRCS})

# 4. 链接全局依赖（OpenCV等，通过根目录的deploy_global传递）
target_link_libraries(deploy_core PRIVATE deploy_global)

# 5. 添加项目内部头文件路径（src目录下的实现头文件）
target_include_directories(deploy_core PRIVATE
  ${PROJECT_SOURCE_DIR}/src  # 让src下的文件能互相引用（如#include "core/Device.hpp"）
)

# 6. 传递公共头文件路径（如果有暴露给外部的接口头文件，通常在include目录）
target_include_directories(deploy_core PUBLIC
  ${PROJECT_SOURCE_DIR}/include  # 让examples等外部目录能引用接口（如#include "infer_engine/Device.hpp"）
)