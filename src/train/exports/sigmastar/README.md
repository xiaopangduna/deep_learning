# 星辰半导体 SigmaStar (SGS IPU)

官方工具链与 Docker **不纳入 Git**。发布包可暂存于 `tmp/IPU_SDK_Release_*`，或放到 `exports/sigmastar/sdk/`（已被 ignore）。

| 项 | 说明 |
|----|------|
| Docker 镜像 | `sgs_docker:v1.8`（Ubuntu 18.04 + Python 3.8） |
| 工具链目录名 | `SGS_IPU_Toolchain_25121210` |
| IDE | 勿用 Dev Container 进该镜像（glibc 2.27 < Cursor Server 所需 2.28）；宿主机 Cursor 编辑 + 容器命令行转换 |

## 1. 加载 Docker 镜像

```bash
cd tmp/IPU_SDK_Release_*/docker_v1.8
tar -Jxvf sgs_docker_v1.8.tar.xz
docker load < sgs_docker_v1.8.tar
./run_docker.sh
```

`run_docker.sh` 将 `$HOME/project/deep_learning` 挂到容器内 `/work/SGS_V1.8_18.04`。

## 2. 初始化工具链

在容器内：

```bash
cd /work/SGS_V1.8_18.04/src/train/tmp/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210
source cfg_env.sh
python3 DumpDebug/show_sdk_info.py   # 查看芯片 / soc_version
```

## 3. Quick Start（YOLOv8s → pcupid）

使用 **`SGS_converter.py`**（不是 `ConvertTool.py`）：

```bash
cd /work/SGS_V1.8_18.04/src/train/tmp/IPU_SDK_Release_25121210/Quick_Start_Demo/onnx_yolov8s

python3 ../../SGS_IPU_Toolchain_25121210/Scripts/ConvertTool/SGS_converter.py onnx \
  -i coco2017_calibration_set32 \
  --model_file onnx_yolov8s.onnx \
  --input_shape 1,3,640,640 \
  --input_config input_config.ini \
  -n onnx_yolov8s_preprocess.py \
  --output_file onnx_yolov8s_pcupid.img \
  --export_models \
  --soc_version pcupid
```

`pcupid` 对应 S03；其它芯片名以 `show_sdk_info.py` / `doc/SDK_Doc_Release/readme.txt` 为准。

## 4. 本项目模型目录（约定）

后续自有模型转换配置放：

```
exports/sigmastar/<模型名>/convert_model/
```

仅提交 README / 脚本 / `*.ini` 等白名单文件；`.onnx` / `.img` / `.sim` 本地生成、不进 Git。
