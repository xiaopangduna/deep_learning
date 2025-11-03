
# add_subdirectory(app)
add_subdirectory(common)
add_subdirectory(infer_engine)
add_subdirectory(processing)
# add_subdirectory(perception_model)
add_subdirectory(utils)


# 可选：一个聚合目标（把三个库安装到一起）
add_library(deploy_core INTERFACE)
target_link_libraries(deploy_core INTERFACE
  common
  infer_engine
  # perception_model
  processing
  utils
)