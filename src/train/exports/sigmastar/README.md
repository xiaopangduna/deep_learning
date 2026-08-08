# 星辰半导体 SigmaStar (SGS IPU)

端侧 IPU 模型转换工具链。`sdk/` 为原厂发布包复制件，**不纳入 Git**。

## 参考资料

| 文档 | 说明 |
|------|------|
| [Comake SDK 下载](https://www.comake.online/support/sdk/) | 开发板资料与 IPU ToolChain 申请下载 |
| [IPU 开发文档](https://doc.comake.online/SGS_IPU_Toolchain_25111209-S21_zh/index.html) | 总览（精度调试、信息查看等） |
| [环境搭建](https://doc.comake.online/SGS_IPU_Toolchain_25111209-S21_zh/module/0_Env_Construction/Env_Construction.html) | Docker 环境 |
| [Quick Start](https://doc.comake.online/SGS_IPU_Toolchain_25111209-S21_zh/module/1_Quick_Start/Quick_Start.html) | 快速转换与仿真 |

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

## 示例

| 示例 | 路径 | 文档 |
|------|------|------|
| YOLOv8 检测 | `Quick_Start_Demo/` | [Quick Start](https://doc.comake.online/SGS_IPU_Toolchain_25111209-S21_zh/module/1_Quick_Start/Quick_Start.html) |

## 支持的模型

`OpenDLAModel_20250721/` 按任务类型划分（验证情况待补充）：

| 目录 | 任务 |
|------|------|
| `classification/` | 分类 |
| `detection/` | 检测 |
| `segment/` | 分割 |
| `pose/` | 姿态 |
| `ocr/` | OCR |
| `asr/` `tts/` `vad/` `speaker/` `sed/` `separation/` | 语音 |
| `llm/` `vlm/` | 大模型 / 多模态 |

本项目自有转换配置放：`exports/sigmastar/<模型名>/`。
