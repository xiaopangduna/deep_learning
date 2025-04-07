# GCC support can be specified at major, minor, or micro version
# (e.g. 8, 8.2 or 8.2.0).
# See https://hub.docker.com/r/library/gcc/ for all supported GCC
# tags from Docker Hub.
# See https://docs.docker.com/samples/library/gcc/ for more on how to use this image
FROM ubuntu:20.04

# These commands copy your files into the specified directory in the image
# and set that as the working location
COPY . /usr/src/myapp
WORKDIR /usr/src/myapp

# This command compiles your app using GCC, adjust for your source code
RUN g++ -o myapp main.cpp

# This command runs your application, comment out this line to compile only
CMD ["./myapp"]

LABEL Name=deploy Version=0.0.1



# # 使用官方的 Ubuntu 镜像作为基础镜像
# FROM ubuntu:20.04


# # 避免交互式安装时的提示
# ARG DEBIAN_FRONTEND=noninteractive

# # 更新包列表并安装构建工具和依赖
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     git \
#     libyaml-dev \
#     && apt-get clean

# # 克隆 yaml-cpp 源代码
# RUN git clone https://github.com/jbeder/yaml-cpp.git /yaml-cpp

# # 创建构建目录并进入
# RUN mkdir /yaml-cpp/build && cd /yaml-cpp/build \
#     # 使用 cmake 配置项目
#     && cmake .. -DYAML_BUILD_SHARED_LIBS=ON \
#     # 编译并安装
#     && make -j$(nproc) \
#     && make install \
#     # 清理构建文件
#     && make clean

# # 设置工作目录
# WORKDIR /app

# # 将当前目录的内容复制到容器内的 /app 目录
# COPY . /app

# # # 暴露容器的端口（根据您的应用需求）
# # EXPOSE 8080

# # 设置容器启动时执行的命令
# CMD ["bash"]
