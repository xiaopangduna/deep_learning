# 全志 Allwinner (awnn)

| 项 | 说明 |
|----|------|
| SDK 路径 | `exports/awnn/sdk/`（原厂 ai-sdk 根目录的复制） |
| 体积 | 约 2.3 GB（解压后） |
| 分支 | `product-aiot-stable` |
| 官方文档 | http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-4-Pro.html |
| Dev Container | `.devcontainer/convert-ubuntu-npu-allwinner`（镜像 `ubuntu-npu:v2.0.10.2`） |

## 准备 SDK

```bash
# 1) 下载/解压到 tmp（示例路径按实际调整）
# 2) 复制到约定位置
cp -a tmp/<ai-sdk解压目录>/. exports/awnn/sdk/
```

```bash
cd exports/awnn/sdk/models
source env.sh v3
cp ../scripts/* .
```

## 本项目内容

| 路径 | 说明 |
|------|------|
| `scripts_model_convert/` | Pegasus 转换脚本模板 |
| `mobilenet_v2/convert_model/` | MobileNetV2 转换示例与说明 |
