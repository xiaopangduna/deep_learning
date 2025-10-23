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



deploy/                                   # project root (/home/orangepi/HectorHuang/deep_learning/src/deploy)
├── CMakeLists.txt
├── README.md
├── LICENSE
├── Dockerfile
├── configs/
│   ├── yolov5_seg_fall_detection.yaml
│   └── preprocess_examples.yaml
├── cmake/
│   ├── DeployConfig.cmake.in
│   ├── toolchain_aarch64.cmake
│   └── FindRKNN.cmake
├── include/                              # public headers (install/export)
│   ├── common/
│   │   ├── Platform.hpp
│   │   ├── DeviceManager.hpp
│   │   ├── Tensor.hpp          # TensorDesc / TensorBuffer / Allocator interfaces
│   │   └── Status.hpp
│   ├── processing/             # processing (pre/post) public API
│   │   ├── Transform.hpp
│   │   ├── Pipeline.hpp
│   │   ├── ImagePipeline.hpp
│   │   ├── PostprocessPipeline.hpp
│   │   └── config/PreprocessConfig.hpp
│   └── infer_engine/
│       ├── InferenceEngine.hpp
│       └── ModelInfo.hpp
├── src/
│   ├── CMakeLists.txt
│   ├── common/
│   │   ├── Platform.cpp
│   │   └── Tensor.cpp
│   ├── processing/              # processing module (pre + post)
│   │   ├── CMakeLists.txt
│   │   ├── core/                # pipeline runner, factory, memory pool, context
│   │   │   ├── PipelineRunner.cpp
│   │   │   └── MemoryPool.cpp
│   │   ├── transforms/          # generic transforms (cpu baseline, post decode)
│   │   │   ├── Resize.cpp
│   │   │   ├── Normalize.cpp
│   │   │   ├── Permute.cpp
│   │   │   └── DecodeYolo.cpp
│   │   ├── cpu/                 # CPU optimized kernels (NEON optional)
│   │   │   └── CPUKernels.cpp
│   │   └── cuda/                # optional CUDA implementations (ENABLE_CUDA)
│   │       └── CudaKernels.cu
│   ├── infer_engine/            # inference engines & backends (optional heavy deps)
│   │   ├── CMakeLists.txt
│   │   ├── platform/
│   │   │   ├── rknn/
│   │   │   │   ├── RknnEngine.cpp
│   │   │   │   └── RknnEngine.hpp
│   │   │   ├── tensorrt/
│   │   │   │   └── TensorRtEngine.cpp
│   │   │   └── mock/
│   │   │       └── MockEngine.cpp
│   │   └── EngineFactory.cpp
│   └── perception_model/
│       ├── Yolov5SegModel.cpp
│       └── PerceptionModelFactory.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── processing/
│   │   └── test_preprocessor.cpp
│   └── infer_engine/
│       └── test_mock_engine.cpp
├── examples/
│   ├── preprocess_demo/
│   │   ├── CMakeLists.txt
│   │   └── demo_pipeline.cpp
│   └── yolov5_seg_example/
│       ├── CMakeLists.txt
│       └── demo_infer.cpp
├── third_party/                  # optional deps (OpenCV, gtest, fmt, etc.)
└── .vscode/
    └── c_cpp_properties.json