# 星辰半导体 SigmaStar (SGS IPU)

端侧 IPU 模型转换工具链。`sdk/` 为原厂发布包复制件，**不纳入 Git**。

## 参考资料

| 文档 | 说明 |
|------|------|
| [Comake SDK 下载](https://www.comake.online/support/sdk/) | 开发板资料与 IPU ToolChain 申请下载 |
| [IPU 开发文档](https://doc.comake.online/SGS_IPU_Toolchain_25111209-S21_zh/index.html) | 总览（精度调试、信息查看等） |
| [环境搭建](https://doc.comake.online/SGS_IPU_Toolchain_25111209-S21_zh/module/0_Env_Construction/Env_Construction.html) | Docker 环境 |
| [Quick Start](https://doc.comake.online/SGS_IPU_Toolchain_25111209-S21_zh/module/1_Quick_Start/Quick_Start.html) | 快速转换与仿真 |
| [OpenDLA/onnx](https://doc.comake.online/IPU_Sigdoc_zh/module/OpenDLA/OpenDLA.html) | ONNX 导出要求、转换与仿真示例 |

## 下载

1. 打开 [Comake SDK 下载页](https://www.comake.online/support/sdk/)
2. 按开发板选择 **Comake Pi D1** 或 **Comake Pi D2**
3. 选择 **Comake_D1_IPU_ToolChain**（Toolchain + Docker v1.8 + OpenDLAModel）
4. **实名认证**后下载，放到：

```text
exports/sigmastar/sdk/IPU_SDK_Release_<版本号>.rar
```



## sdk/ 内容说明

解压后结构（以 `IPU_SDK_Release_25121210` 为例，仅最外层）：

```text
exports/sigmastar/sdk/IPU_SDK_Release_25121210/
├── docker_v1.8/                  # 转换环境（sgs_docker:v1.8）
├── Quick_Start_Demo/             # 快速示例（如 YOLOv8 onnx → img）
├── SGS_IPU_Toolchain_25121210/   # 工具链主体（Scripts / libs / cfg_env.sh）
└── OpenDLAModel_20250721/        # 分任务类型的实战示例
```



## 流程



### 1. 解压

```bash
cd exports/sigmastar/sdk
# 可选：sudo apt install unrar
unrar x IPU_SDK_Release_25121210.rar
cd IPU_SDK_Release_25121210
tar -xvf SGS_IPU_Toolchain_25121210.tar.xz
tar -xvf OpenDLAModel_20250721.tar
```



### 2. 加载并启动容器

先按需修改 `docker_v1.8/run_docker.sh` 的 `-v`（建议挂载 `$HOME/project/deep_learning`）。

```bash
cd docker_v1.8
tar -Jxvf sgs_docker_v1.8.tar.xz
docker load < sgs_docker_v1.8.tar
./run_docker.sh
```



### 3. 容器内初始化工具链

工程一般在 `/work/SGS_V1.8_18.04/`（与容器名一致）：

```bash
cd /work/SGS_V1.8_18.04/src/train/exports/sigmastar/sdk/IPU_SDK_Release_*/SGS_IPU_Toolchain_*
source cfg_env.sh
```

`cfg_env.sh` 会设置 `SGS_IPU_DIR`、`PYTHONPATH`，并把 `libs/x86_64` 加入 `LD_LIBRARY_PATH`（缺了会报 `libLLVM-16.so: cannot open shared object file`）。

## 模型转换

建议日常用 **OpenDLAModel 调度脚本**，不要直接手填子目录 `convert.sh` 的 9 个位置参数。

```text
OpenDLAModel_20250721/convert.sh          # 调度：读 cfg、source 工具链、按模型循环
  ├─ detection/yolov8/convert.sh          # CPU 后处理：ONNX → float.sim → fixed.sim → .img
  └─ detection/yolov8/convert_ipu.sh      # IPU 后处理：额外挂 yolov8_post.py（NMS 等进 IPU）
```

三步含义：


| 步骤        | 工具                    | 产物            |
| --------- | --------------------- | ------------- |
| ONNX → 浮点 | `ConvertTool.py`      | `*_float.sim` |
| 浮点 → 定点   | `calibrator.py`（需校准图） | `*_fixed.sim` |
| 定点 → 离线   | `compiler.py`         | `*.img`（板端）   |




### YOLOv8（推荐）

`-s`（转换 / 仿真）和 `-d`（CPU / IPU 后处理）不要混用：`-d ipu` 会丢掉 `-s`。

在容器内先 `cd` 到 OpenDLA 目录：

```bash
cd /work/SGS_V1.8_18.04/src/train/exports/sigmastar/sdk/IPU_SDK_Release_25121210/OpenDLAModel_20250721
```

转换（二选一）：

```bash
# CPU 后处理
bash convert.sh -a detection/yolov8 -c config/detection_yolov8.cfg -p ../SGS_IPU_Toolchain_25121210 -s false

# IPU 后处理（NMS 进 IPU；不要再写 -s）
bash convert.sh -a detection/yolov8 -c config/detection_yolov8.cfg -p ../SGS_IPU_Toolchain_25121210 -d ipu
```

仿真（须先转换，且不要带 `-d ipu`）：

```bash
bash convert.sh -a detection/yolov8 -c config/detection_yolov8.cfg -p ../SGS_IPU_Toolchain_25121210 -s true
```

| 参数 | 含义 |
|------|------|
| `-a` | 算法目录（相对 `OpenDLAModel_20250721/`） |
| `-c` | 转换配置，默认芯片 `pcupid`、模型 `yolov8n` |
| `-p` | 工具链真实路径（相对或绝对均可） |
| `-s false` | CPU 路径：转换 |
| `-s true` | CPU 路径：仿真已有 `*_float.sim` |
| `-d ipu` | IPU 路径：只转换，忽略 `-s` |

`config/detection_yolov8.cfg` 会指定模型名、输入尺寸、校准数据目录、输出 `.img` 名。转换前确认：

- `detection/yolov8/onnx/yolov8n.onnx`
- `detection/yolov8/quant_data/`（校准图）

转换产物：`OpenDLAModel_20250721/output/pcupid_<日期>/`，例如 `yolov8n_640x640.img`。  
仿真产物：`detection/yolov8/log/output/unknown_yolov8n_float.sim_<图片名>.txt`（同一天目录名才对得上 float.sim）。

本项目自有转换配置放 `exports/sigmastar/<模型名>/`，不要改 `sdk/` 里的原厂示例。

### 一键转换（Quick Start）

适合先打通流程。须先 `source cfg_env.sh`，在 `IPU_SDK_Release_25121210/` 下执行：

```bash
python3 SGS_IPU_Toolchain_25121210/Scripts/ConvertTool/SGS_converter.py onnx \
  -i Quick_Start_Demo/onnx_yolov8s/coco2017_calibration_set32/ \
  --model_file Quick_Start_Demo/onnx_yolov8s/<your>.onnx \
  --input_shape 1,3,640,640 \
  --input_config Quick_Start_Demo/onnx_yolov8s/input_config.ini \
  -n Quick_Start_Demo/onnx_yolov8s/onnx_yolov8s_preprocess.py \
  --output_file yolov8_pcupid.img \
  --export_models \
  --postprocess Quick_Start_Demo/onnx_yolov8s/onnx_yolov8s_postprocess.py \
  --soc_version pcupid
```

## 常用指令

```bash
cd .../IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210/
python3 DumpDebug/show_sdk_info.py
python3 DumpDebug/show_img_info.py \
  -m ../Quick_Start_Demo/onnx_yolov8s/onnx_yolov8n_pcupid.img \
  --soc_version pcupid
```

开发板查看模型信息：Linux SDK-alkaid 已提供 `sdk/verify/release_feature/source/dla/dla_dla_show_img_info`。

## 模型推理

- PC 仿真：`Quick_Start_Demo/onnx_yolov8s/yolov8_simulator.py`，或 OpenDLA 的 `-s true`
- 开发板：RPC 跑 `.img`



## 示例


| 示例        | 路径                  | 文档                                                                                                               |
| --------- | ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| YOLOv8 检测 | `Quick_Start_Demo/` | [Quick Start](https://doc.comake.online/SGS_IPU_Toolchain_25111209-S21_zh/module/1_Quick_Start/Quick_Start.html) |




## 支持的模型

`OpenDLAModel_20250721/` 按任务类型划分（验证情况待补充）：


| 目录                                                   | 任务        |
| ---------------------------------------------------- | --------- |
| `classification/`                                    | 分类        |
| `detection/`                                         | 检测        |
| `segment/`                                           | 分割        |
| `pose/`                                              | 姿态        |
| `ocr/`                                               | OCR       |
| `asr/` `tts/` `vad/` `speaker/` `sed/` `separation/` | 语音        |
| `llm/` `vlm/`                                        | 大模型 / 多模态 |


