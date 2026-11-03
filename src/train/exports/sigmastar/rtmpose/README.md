# RTMPose 人脸关键点转换

容器内在本目录执行。参数与 OpenDLA 调度脚本一致，省略 `-a` 和 `-d`：本目录就是模型目录，关键点后处理不进 IPU。

```bash
cd exports/sigmastar/rtmpose

# 灰度 rtmpose，输入 1x1x256x256，产物 rtmpose_gray_42.img
bash convert.sh \
  -c config/landmark_gray.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s false

# 仿真已有的浮点模型。须先转换，且对得上当天的 release 目录
bash convert.sh \
  -c config/landmark_gray.cfg \
  -p ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210 \
  -s true
```

`-p` 不写时用上面这条相对路径。`-s` 不写时为 `false`。产物在 `release/<芯片>_<日期>/`。

预处理与原来的 `RTMPose_pre.py` 一致：PIL 转灰度、不缩放，像素做 `(x - 107.58) / 57.94`，输出 NHWC。`input_config_gray.ini` 里是同一组 `mean` / `std_value`，输入名 `input`，输出 `simcc_x, simcc_y`。换模型时改 cfg 的 `Model_LIST` 和 `SAVE_NAME_LIST`，并把 ONNX 放到 `onnx/`。
