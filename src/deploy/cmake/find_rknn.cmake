find_path(TensorRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/local/TensorRT/include /opt/TensorRT/include)
find_library(TensorRT_LIBRARY nvinfer
  PATHS /usr/local/TensorRT/lib /opt/TensorRT/lib)
# ... 检查版本、其他库（nvonnxparser等）