#!/bin/bash
# YOLOv8 检测转换。在本目录执行。
# 用法接近 OpenDLA 调度脚本，但不接收 -a：目录已是模型目录。
# -d cpu 后处理在 CPU；-d ipu 把 NMS 放进 IPU。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
DEFAULT_TOOLCHAIN="$ROOT/../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210"

CONVERT_INI=""
TOOLCHAIN="$DEFAULT_TOOLCHAIN"
SIMULATOR=false
DECODE=cpu

usage() {
	cat <<EOF
Usage: bash convert.sh -c <cfg> [-p toolchain] [-s true|false] [-d cpu|ipu]

  -c  转换配置。COCO config/detection_yolov8.cfg，头检测 config/detection_yolov8_head.cfg
  -p  SGS_IPU_Toolchain 目录。默认 ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210
  -s  false 转换（默认）；true 只仿真已有的 *_float.sim。不能和 -d ipu 一起用
  -d  cpu 后处理在 CPU（默认）；ipu 把 NMS 放进 IPU，只转换

产物在 release/<芯片>_<日期>/。工具链会占用本目录的 output/ 和 log/，不要把模型写到那里。
EOF
}

while getopts ":c:p:s:d:h" opt; do
	case "$opt" in
		c) CONVERT_INI=$OPTARG ;;
		p) TOOLCHAIN=$OPTARG ;;
		s) SIMULATOR=$OPTARG ;;
		d) DECODE=$OPTARG ;;
		h) usage; exit 0 ;;
		\?) echo "Unknown option: -$OPTARG" >&2; usage; exit 1 ;;
		:) echo "Option -$OPTARG requires a value" >&2; usage; exit 1 ;;
	esac
done

if [ -z "$CONVERT_INI" ]; then
	echo "-c is required" >&2
	usage
	exit 1
fi
case "$SIMULATOR" in
	true|false) ;;
	*) echo "-s must be true or false, got: $SIMULATOR" >&2; exit 1 ;;
esac
case "$DECODE" in
	cpu|ipu) ;;
	*) echo "-d must be cpu or ipu, got: $DECODE" >&2; exit 1 ;;
esac
if [ "$DECODE" = ipu ] && [ "$SIMULATOR" = true ]; then
	echo "-d ipu only converts; do not pass -s true" >&2
	exit 1
fi

if [[ "$CONVERT_INI" != /* && ! -f "$CONVERT_INI" && -f "$ROOT/$CONVERT_INI" ]]; then
	CONVERT_INI="$ROOT/$CONVERT_INI"
fi
if [ ! -f "$CONVERT_INI" ]; then
	echo "Missing config: $CONVERT_INI" >&2
	exit 1
fi

cfg_value() {
	grep "$1" "$CONVERT_INI" | awk -F '=' '{print $2}' | tr -d '[:space:]'
}

MODEL=$(cfg_value 'Model_LIST')
INPUT_SIZE=$(cfg_value 'INPUT_SIZE_LIST')
INPUT_INI=$(cfg_value 'INPUT_INI_LIST')
CLASS_NUM=$(cfg_value 'CLASS_NUM_LIST')
CHIP=$(cfg_value 'CHIP_LIST')
SAVE_NAME=$(cfg_value 'SAVE_NAME_LIST')
QUANT_DATA=$(cfg_value 'QUANT_DATA_PATH')

for pair in \
	"Model_LIST:$MODEL" \
	"INPUT_SIZE_LIST:$INPUT_SIZE" \
	"INPUT_INI_LIST:$INPUT_INI" \
	"CLASS_NUM_LIST:$CLASS_NUM" \
	"CHIP_LIST:$CHIP" \
	"SAVE_NAME_LIST:$SAVE_NAME" \
	"QUANT_DATA_PATH:$QUANT_DATA"
do
	key=${pair%%:*}
	value=${pair#*:}
	if [ -z "$value" ]; then
		echo "Missing $key in $CONVERT_INI" >&2
		exit 1
	fi
	case "$value" in
		*,*)
			echo "$key lists more than one value: $value" >&2
			exit 1
			;;
	esac
done

IFS='x' read -ra DIMS <<< "${INPUT_SIZE}"
case ${#DIMS[@]} in
	4)
		INPUT_SHAPES="${DIMS[0]},${DIMS[1]},${DIMS[2]},${DIMS[3]}"
		YOLO_IN_CHANNELS=${DIMS[1]}
		YOLO_IN_HEIGHT=${DIMS[2]}
		YOLO_IN_WIDTH=${DIMS[3]}
		;;
	2)
		INPUT_SHAPES="1,3,${DIMS[0]},${DIMS[1]}"
		YOLO_IN_CHANNELS=3
		YOLO_IN_HEIGHT=${DIMS[0]}
		YOLO_IN_WIDTH=${DIMS[1]}
		;;
	*)
		echo "Unsupported INPUT_SIZE_LIST=${INPUT_SIZE}, expect HxW or NxCxHxW" >&2
		exit 1
		;;
esac
if [ "$CLASS_NUM" -gt 0 ] 2>/dev/null; then
	YOLO_NUM_CLASSES=$CLASS_NUM
else
	YOLO_NUM_CLASSES=80
fi
export YOLO_IN_CHANNELS YOLO_IN_HEIGHT YOLO_IN_WIDTH YOLO_NUM_CLASSES

PREPROCESS="$ROOT/yolov8_pre.py"
POSTPROCESS="$ROOT/yolov8_post.py"

ONNX="$ROOT/onnx/${MODEL}.onnx"
if [[ "$INPUT_INI" != /* ]]; then
	INI="$ROOT/$INPUT_INI"
else
	INI="$INPUT_INI"
fi
if [[ "$QUANT_DATA" != /* ]]; then
	QUANT="$ROOT/$QUANT_DATA"
else
	QUANT="$QUANT_DATA"
fi
if [[ "$TOOLCHAIN" != /* ]]; then
	TOOLCHAIN="$(cd "$(dirname "$TOOLCHAIN")" && pwd)/$(basename "$TOOLCHAIN")"
fi
OFFLINE_INI="$ROOT/new_offline_$(basename "$INI")"

DATE=$(date +%Y%m%d)
# calibrator.py 会删除 cwd 下的 output/，模型产物不能放在那里。
SAVE_DIR="$ROOT/release/${CHIP}_${DATE}"

echo "convert parameters:
	Toolchain: ${TOOLCHAIN}
	Config: ${CONVERT_INI}
	Model: ${MODEL}
	InputSize: ${INPUT_SIZE} -> ${INPUT_SHAPES}
	ClassNum: ${YOLO_NUM_CLASSES}
	Preprocess: ${PREPROCESS}
	InputIni: ${INI}
	QuantData: ${QUANT}
	Chip: ${CHIP}
	Decode: ${DECODE}
	Simulator: ${SIMULATOR}
	SaveDir: ${SAVE_DIR}
	SaveName: ${SAVE_NAME}"

for f in "$ONNX" "$INI" "$PREPROCESS" "$TOOLCHAIN/cfg_env.sh" "$TOOLCHAIN/Scripts/ConvertTool/ConvertTool.py"; do
	if [ ! -e "$f" ]; then
		echo "Missing: $f" >&2
		exit 1
	fi
done
if [ "$DECODE" = ipu ]; then
	for f in "$POSTPROCESS" "$OFFLINE_INI" "$ROOT/modify_out_nodes.py"; do
		if [ ! -e "$f" ]; then
			echo "Missing: $f" >&2
			exit 1
		fi
	done
fi
if [ ! -d "$QUANT" ]; then
	echo "Missing calibration dir: $QUANT" >&2
	exit 1
fi

mkdir -p "$SAVE_DIR"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
cd "$TOOLCHAIN"
# cfg_env.sh 用当前目录作为 SGS_IPU_DIR，且会读取可能未定义的 PYTHONPATH
set +u
source cfg_env.sh
set -u
cd "$ROOT"

if [ "$DECODE" = ipu ]; then
	# 改输出节点会重写 ini。写到 release 目录，避免改掉本目录里的配置。
	WORK_INI="${SAVE_DIR}/$(basename "$INI")"
	WORK_POST="${SAVE_DIR}/yolov8_post.py"
	cp "$INI" "$WORK_INI"
	cp "$POSTPROCESS" "$WORK_POST"
	python3 "$ROOT/modify_out_nodes.py" \
		-n output0 \
		-i "$WORK_INI" \
		-p "$WORK_POST"

	echo "convert onnx model to float model"
	python3 "${SGS_IPU_DIR}/Scripts/ConvertTool/ConvertTool.py" onnx \
		--model_file "$ONNX" \
		--input_config "$WORK_INI" \
		--input_shapes "$INPUT_SHAPES" \
		--output_file "${SAVE_DIR}/${MODEL}_float.sim" \
		--postprocess "$WORK_POST" \
		--postprocess_input_config "$OFFLINE_INI" \
		--soc_version "$CHIP"

	echo "convert float model to fixed model"
	python3 "${SGS_IPU_DIR}/Scripts/calibrator/calibrator.py" \
		--input_config "$OFFLINE_INI" \
		-i "$QUANT" \
		-m "${SAVE_DIR}/${MODEL}_float.sim" \
		-n "$PREPROCESS" \
		--soc_version "$CHIP"

	echo "convert fixed model to offline model"
	python3 "${SGS_IPU_DIR}/Scripts/calibrator/compiler.py" \
		--input_config "$OFFLINE_INI" \
		-m "${SAVE_DIR}/${MODEL}_fixed.sim" \
		--output "${SAVE_DIR}/${SAVE_NAME}" \
		--soc_version "$CHIP"
elif [ "$SIMULATOR" = false ]; then
	echo "convert onnx model to float model"
	python3 "${SGS_IPU_DIR}/Scripts/ConvertTool/ConvertTool.py" onnx \
		--model_file "$ONNX" \
		--input_config "$INI" \
		--input_shapes "$INPUT_SHAPES" \
		--output_file "${SAVE_DIR}/${MODEL}_float.sim" \
		--soc_version "$CHIP"

	echo "convert float model to fixed model"
	python3 "${SGS_IPU_DIR}/Scripts/calibrator/calibrator.py" \
		--input_config "$INI" \
		-i "$QUANT" \
		-m "${SAVE_DIR}/${MODEL}_float.sim" \
		-n "$PREPROCESS" \
		--soc_version "$CHIP"

	echo "convert fixed model to offline model"
	python3 "${SGS_IPU_DIR}/Scripts/calibrator/compiler.py" \
		-m "${SAVE_DIR}/${MODEL}_fixed.sim" \
		--output "${SAVE_DIR}/${SAVE_NAME}" \
		--soc_version "$CHIP"
else
	python3 "${SGS_IPU_DIR}/Scripts/calibrator/simulator.py" \
		-i "$QUANT" \
		-c Unknown \
		-t Float \
		-n "$PREPROCESS" \
		-m "${SAVE_DIR}/${MODEL}_float.sim" \
		--soc_version "$CHIP"
fi
