
if(ENABLE_RKNN)
  add_subdirectory(rknn)
endif()

if(ENABLE_ONNXRUNTIME)
  add_subdirectory(onnxruntime)
endif()