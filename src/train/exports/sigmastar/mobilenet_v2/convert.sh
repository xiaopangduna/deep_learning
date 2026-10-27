#!/bin/bash
# MobileNetV2 分类转换。在本目录执行。
# 用法接近 OpenDLA 调度脚本，但不接收 -a / -d：目录已是模型目录，分类后处理不进 IPU。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
DEFAULT_TOOLCHAIN="$ROOT/../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210"

CONVERT_INI=""
TOOLCHAIN="$DEFAULT_TOOLCHAIN"
SIMULATOR=false

usage() {
	cat <<EOF
Usage: bash convert.sh -c <cfg> [-p toolchain] [-s true|false]

  -c  转换配置。灰度 config/classification_gray.cfg，RGB config/classification.cfg
  -p  SGS_IPU_Toolchain 目录。默认 ../sdk/IPU_SDK_Release_25121210/SGS_IPU_Toolchain_25121210
  -s  false 转换（默认）；true 只仿真已有的 *_float.sim

产物在 release/<芯片>_<日期>/。工具链会占用本目录的 output/ 和 log/，不要把模型写到那里。
EOF
}

while getopts ":c:p:s:h" opt; do
	case "$opt" in
		c) CONVERT_INI=$OPTARG ;;
		p) TOOLCHAIN=$OPTARG ;;
		s) SIMULATOR=$OPTARG ;;
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
CHIP=$(cfg_value 'CHIP_LIST')
SAVE_NAME=$(cfg_value 'SAVE_NAME_LIST')
QUANT_DATA=$(cfg_value 'QUANT_DATA_PATH')

for pair in \
	"Model_LIST:$MODEL" \
	"INPUT_SIZE_LIST:$INPUT_SIZE" \
	"INPUT_INI_LIST:$INPUT_INI" \
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
if [ "${#DIMS[@]}" -ne 4 ]; then
	echo "Unsupported INPUT_SIZE_LIST=${INPUT_SIZE}, expect NxCxHxW like 1x1x224x224" >&2
	exit 1
fi
INPUT_SHAPES="${DIMS[0]},${DIMS[1]},${DIMS[2]},${DIMS[3]}"
CLS_IN_CHANNELS=${DIMS[1]}
CLS_IN_HEIGHT=${DIMS[2]}
CLS_IN_WIDTH=${DIMS[3]}
export CLS_IN_CHANNELS CLS_IN_HEIGHT CLS_IN_WIDTH

if [ "$CLS_IN_CHANNELS" = "1" ]; then
	PREPROCESS="$ROOT/mobilenet_gray_pre.py"
else
	PREPROCESS="$ROOT/mobilenet_pre.py"
fi

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

DATE=$(date +%Y%m%d)
# calibrator.py 会删除 cwd 下的 output/，模型产物不能放在那里。
SAVE_DIR="$ROOT/release/${CHIP}_${DATE}"

echo "convert parameters:
	Toolchain: ${TOOLCHAIN}
	Config: ${CONVERT_INI}
	Model: ${MODEL}
	InputSize: ${INPUT_SIZE} -> ${INPUT_SHAPES}
	Preprocess: ${PREPROCESS}
	InputIni: ${INI}
	QuantData: ${QUANT}
	Chip: ${CHIP}
	Simulator: ${SIMULATOR}
	SaveDir: ${SAVE_DIR}
	SaveName: ${SAVE_NAME}"

for f in "$ONNX" "$INI" "$PREPROCESS" "$TOOLCHAIN/cfg_env.sh" "$TOOLCHAIN/Scripts/ConvertTool/ConvertTool.py"; do
	if [ ! -e "$f" ]; then
		echo "Missing: $f" >&2
		exit 1
	fi
done
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

if [ "$SIMULATOR" = false ]; then
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
