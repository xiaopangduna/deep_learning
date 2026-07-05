# 概述

本文档描述mobilenetv2模型在NPU的部署过程，含模型转换与板端示例两部分，其中`convert_model`为模型转换的目录，其它文件为板端运行的示例代码等文件。



# 获取onnx模型

开源模型地址：

https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet

下载mobilenetv2-12.onnx模型，将下载的onnx模型保存到convert_model目录。



# 模型转换

模型转换主要包含原始模型导入、量化、导出为NPU可识别的模型格式等步骤。

```bash
cd ./convert_model/
```

修改config_yml.py文件的相关参数配置；

```python
# "database"
DATASET = '../../dataset/imagenet_10/dataset.txt'
DATASET_TYPE = "TEXT"
# mean, scale
MEAN    = [123.675, 116.28, 103.53]
SCALE   = [0.0171247, 0.0175070, 0.0174291]

# reverse_channel: True bgr, False rgb
REVERSE_CHANNEL = False
# add_preproc_node, True or False
ADD_PREPROC_NODE = True
# "preproc_type" 
PREPROC_TYPE = "IMAGE_RGB"
# add_postproc_node, quant output -> float32 output
ADD_POSTPROC_NODE = True
```

模型导入、量化、导出等步骤：

```bash
# using xxx_env.sh to create softlink
./convert_model_env.sh

# 导入
# pegasus_import.sh <model_name>
./pegasus_import.sh mobilenetv2-12
 
# 量化
# pegasus_quantize.sh <model_name> <quantize_type> <calibration_set_size>
./pegasus_quantize.sh mobilenetv2-12 pcq 10

# 仿真（可选）
# pegasus_inference.sh <model_name> <quantize_type>
./pegasus_inference.sh mobilenetv2-12 pcq

# 导出nb模型
# pegasus_export_ovx_nbg.sh <model_name> <quantize_type> <platform>
./pegasus_export_ovx_nbg.sh mobilenetv2-12 pcq t527

# 导出的模型文件存放在../model目录
# 例如 ../model/mobilenetv2-12_pcq_t527.nb
```



# 板端demo

含demo编译及运行说明。



## 解压opencv压缩包

```bash
# 进入目录
cd ../../../3rdparty/opencv/
# 解压，选择对应平台
# armhf, eg: V85x, R853
unzip opencv-3.4.16-gnueabihf-linux.zip
# linux aarch64, eg: T527/MR527/MR536/T536/A733/T736
unzip opencv-4.9.0-aarch64-linux-sunxi-glibc.zip
# android aarch64, eg: T527/A733/T736
unzip opencv-4.9.0-android.zip
```



## 准备交叉编译工具链

### Linux

```bash
# 进入目录
cd ../../0-toolchains/
# 解压
# armhf, V85x, R853
unzip arm-openwrt-linux-muslgnueabi.zip
chmod 777 -R ./arm-openwrt-linux-muslgnueabi
# aarch64, MR527, T527, MR536, T536, A733, T736
tar xvf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
# aarch64 for debian11, T527, A733, T736
tar vxf gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu.tar.xz
```

编译脚本会根据平台自动选择交叉编译工具链，若需使用其它路径的工具链，可在`cmake_toolchain`目录修改`.cmake`文件内容指定对应的交叉编译工具链路径。



### Android

下载Android NDK，下载地址：https://developer.android.google.cn/ndk/downloads?hl=zh-cn

将下载的NDK放到编译机器目录，例如：./0-toolchains/ ;

请根据下载的版本修改`cmake_toolchain`目录的`android_ndk_build_env.sh` 文件。

使用unzip命令解压。

## build && run

### Linux

在Linux系统下测试。编译用法如下：

```bash
# 途径一：在mobilenetv2目录编译
cd ../examples/mobilenetv2/
./../build_linux.sh -t <platform> [-s <system>]
# 途径二：在examples目录，再选择mobilenetv2目录编译
cd ../examples
./build_linux.sh -t <platform> -p mobilenetv2 [-s <system>]
```

以下说明以T527平台为例；

```bash
cd ../examples/mobilenetv2/
./../build_linux.sh -t t527
```

> 若是T527平台debian系统，则是以下命令：
>
> ```bash
> cd ../examples/mobilenetv2/
> ./../build_linux.sh -t t527 -s debian11
> ```

push 可执行文件、模型文件、输入图片到板端目录（建议推到tf卡目录，空间充足）；
```
adb push .\install\mobilenetv2_demo_linux_t527 /mnt/UDISK/
```

运行；

```bash
adb shell
cd /mnt/UDISK/mobilenetv2_demo_linux_t527

# 可选
export LD_LIBRARY_PATH=./lib

# 运行可执行文件
# ./mobilenetv2_demo_t527 -h 查看执行示例
chmod +x ./mobilenetv2_demo_t527
./mobilenetv2_demo_t527 -nb model/mobilenetv2_pcq_t527.nb -i model/1.jpg
```

运行后，可以看到TOP5结果输出 ;

```bash
========== top5 ==========
class id: 281, prob: 14.504395, label: tabby, tabby cat
class id: 282, prob: 12.977539, label: tiger cat
class id: 285, prob: 12.341553, label: Egyptian cat
class id: 287, prob: 8.015625, label: lynx, catamount
class id: 478, prob: 8.015625, label: carton
```

### Android

在Android 64bit系统下测试。编译用法如下：

```bash
# 途径一：在mobilenetv2目录编译
cd ../examples/mobilenetv2/
./../build_android.sh -t <platform>
# 途径二：在examples目录，再选择mobilenetv2目录编译
cd ../examples
./build_android.sh -t <platform> -p mobilenetv2
```

以下说明以T527平台为例；

```bash
cd ../examples/mobilenetv2/
./../build_android.sh -t t527
```

修改权限；

```bash
adb root
adb remount
```

push 可执行文件、模型文件、输入图片到`/data/local/`目录；

```bash
adb push install\mobilenetv2_demo_android_t527 /data/local/
```

运行；

```bash
adb shell
cd /data/local/mobilenetv2_demo_android_t527

export LD_LIBRARY_PATH=./lib

# 运行可执行文件
chmod +x ./mobilenetv2_demo_t527
./mobilenetv2_demo_t527 -nb model/mobilenetv2_xxx.nb -i model/1.jpg
```
运行后，可以看到TOP5结果输出（同上文）。
