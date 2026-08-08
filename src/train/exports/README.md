# 板端模型转换（exports）

训练项目负责训练与导出（ONNX/PT 等）；本目录存放**各芯片商的转换脚本、配置与说明**。

官方 SDK / Docker 体积大，**不纳入 Git**：可放在 `exports/<厂商>/sdk/`（已被忽略），或仓库外 / `tmp/` 下载暂存。

## 目录约定

```
exports/
  README.md                 # 本说明
  <厂商>/
    README.md               # SDK 获取与转换入口
    scripts/                # 可选：本项目封装脚本
    sdk/                    # 可选：本地 SDK（不进 Git）
    <模型>/convert_model/   # 配置与脚本进 Git；产物本地生成
  quant_data/               # 校准小集（进 Git）
```

## Git 跟踪规则

由 `exports/.gitignore` 管理：**默认忽略，白名单放行**。

| 跟踪 | 不跟踪 |
|------|--------|
| `README.md`、`.gitignore` | `sdk/`、工具链、docker 镜像包 |
| `*.py` / `*.sh` / `*.yml` / `*.yaml` / `*.ini` / `*.txt` | `.onnx` / `.img` / `.sim` / `.nb` / `.data` 等产物 |
| `quant_data/**`（含图片） | `inf/`、`wksp/`、`convert_model/pegasus_*.sh` 软链、`*_inputmeta.yml` 等本地生成 |

原则：只跟踪「怎么转」，不跟踪「用什么转」和「转出来什么」（`quant_data` 例外）。

## 厂商入口

| 厂商 | 路径 |
|------|------|
| 全志 Allwinner (awnn) | [awnn/](awnn/)（脚本见 `scripts_model_convert/`；SDK 见下方） |
| 星辰 SigmaStar (SGS IPU) | [sigmastar/README.md](sigmastar/README.md) |

### 全志 Allwinner SDK

| 项 | 说明 |
|----|------|
| 建议路径 | `exports/awnn/sdk/`（即原 ai-sdk 根目录） |
| 体积 | 约 2.3 GB（解压后） |
| 分支 | `product-aiot-stable` |
| 官方文档 | http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-4-Pro.html |

```bash
# 将官方 ai-sdk 放到 exports/awnn/sdk/ 后：
cd exports/awnn/sdk/models
source env.sh v3
cp ../scripts/* .
```

Dev Container：`.devcontainer/convert-ubuntu-npu-allwinner`（镜像 `ubuntu-npu:v2.0.10.2`）。
