# 模型转换工具（本地自备，不纳入 Git）

训练项目负责训练与导出（ONNX/PT 等）；本目录存放各芯片商的板端转换与部署 SDK，需在本机单独下载。

## 目录约定

```
convert/<芯片商>/<SDK 名称>/
```

## 全志 Allwinner

| 项 | 说明 |
|----|------|
| 路径 | `convert/allwinner/ai-sdk/` |
| 体积 | 约 2.3 GB（解压后） |
| 分支 | `product-aiot-stable` |

### 方式一：

从官方文档中下载http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-4-Pro.html


执行
cd convert/allwinner/ai-sdk/models
source env.sh v3
cp ../scripts/* .