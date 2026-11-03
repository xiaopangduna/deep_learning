# YOLOv8 检测转换

容器内在本目录执行。参数与 OpenDLA 调度脚本一致，省略 `-a`：本目录就是模型目录。`-d cpu`（默认）后处理在 CPU，`-d ipu` 把 NMS 放进 IPU。`-d ipu` 不要再写 `-s`。

```bash
cd exports/sigmastar/yolov8

# CPU 后处理。COCO yolov8n，输入 640x640，芯片 pcupid
bash convert.sh \
  -c config/detection_yolov8.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s false

# IPU 后处理（NMS 进 IPU）
bash convert.sh \
  -c config/detection_yolov8.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -d ipu

# 头检测。灰度 1x1x352x640，2 类，芯片 iford
bash convert.sh \
  -c config/detection_yolov8_head.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s false

# 仿真已有的浮点模型。须先走 CPU 转换，且对得上当天的 release 目录
bash convert.sh \
  -c config/detection_yolov8.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s true
```

`-p` 不写时用上面这条相对路径。`-s` 不写时为 `false`，`-d` 不写时为 `cpu`。产物在 `release/<芯片>_<日期>/`。

`CLASS_NUM_LIST=0` 时类别数按 80。`INPUT_SIZE_LIST` 可以写 `640x640`，也可以写 `1x1x352x640` 这种 NxCxHxW。换模型时改对应 cfg 的 `Model_LIST` 和 `SAVE_NAME_LIST`，并把 ONNX 放到 `onnx/`。
