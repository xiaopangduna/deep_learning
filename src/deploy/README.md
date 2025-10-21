infer_engine/                  # 根目录
├── include/                   # 公共头文件（对外暴露的API）
│   ├── infer_engine/          # 核心接口（如ModelDeployer、InferenceEngine）
│   ├── preprocess/            # 预处理接口
│   ├── postprocess/           # 后处理接口
│   └── utils/                 # 工具类（日志、性能统计）
├── src/                       # 实现代码
│   ├── core/                  # 核心层（模型管理、推理调度）
│   ├── backend/               # 硬件后端实现（按后端分目录）
│   │   ├── onnxruntime/       # ONNX Runtime后端
│   │   ├── tensorrt/          # TensorRT后端
│   │   ├── tflite/            # TFLite后端
│   │   └── ascend/            # 昇腾NPU后端（可选）
│   ├── preprocess/            # 预处理实现（依赖OpenCV等）
│   ├── postprocess/           # 后处理实现（按任务分：检测/分类/分割）
│   └── utils/                 # 工具类实现
├── examples/                  # 示例代码（如yolov8部署、分类示例）
├── cmake/                     # 自定义CMake模块（查找第三方依赖）
│   ├── FindTensorRT.cmake
│   ├── FindAscendCL.cmake
│   └── ...
├── CMakeLists.txt             # 根CMakeLists
└── README.md


用户编译安装后，可在自己的项目中通过以下方式引用：
find_package(infer_engine REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE infer_engine)