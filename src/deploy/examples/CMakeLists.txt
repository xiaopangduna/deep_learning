# examples/CMakeLists.txt
# 遍历所有demo子目录，或显式添加（推荐显式添加，避免意外包含无关目录）

# 显式列出需要编译的demo（便于控制哪些demo参与编译）
set(DEMO_SUBDIRS
  yolov5_seg_example
  # classification_demo
  # detection_demo
)

# 逐个添加子目录（每个子目录的CMakeLists负责自己的编译）
foreach(demo ${DEMO_SUBDIRS})
  add_subdirectory(${demo})
  message(STATUS "Added example: ${demo}")
endforeach()