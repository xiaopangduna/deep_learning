#!/bin/bash
set -e

# 删除旧的build目录（如果存在）
rm -rf build

# 创建并进入新的build目录
mkdir -p build
cd build

# 生成Makefile并编译
#cmake ..
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
make -j4