#!/usr/bin/env bash
# 逐条执行下面的排队。只改「排队从这里往下改」以下的内容。
#
#   bash scripts/run_commands.sh
#
# 每个 step 是一段 shell，在单独的子 shell 里执行。脚本启动时位于仓库根。
# 要换目录就在这段里先 cd，这次 cd 不会带到下一个 step。
# 一段里的命令失败后，这一段停下，下一个 step 继续。
# 结束时列出成功和失败。Ctrl-C 停掉后续 step。
# 正文用单引号包住。注释掉一个 step 时，从 step 到结束引号的每一行都加上 #。

set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || {
  echo "无法进入仓库根" >&2
  exit 1
}

ok=()
bad=()
step_n=0
interrupted=0

summary() {
  echo "========== 汇总 ==========" >&2
  echo "成功 (${#ok[@]}):" >&2
  if [[ ${#ok[@]} -eq 0 ]]; then
    echo "  (无)" >&2
  else
    local item
    for item in "${ok[@]}"; do
      echo "  - $item" >&2
    done
  fi
  echo >&2
  echo "失败 (${#bad[@]}):" >&2
  if [[ ${#bad[@]} -eq 0 ]]; then
    echo "  (无)" >&2
  else
    local item
    for item in "${bad[@]}"; do
      echo "  - $item" >&2
    done
  fi
}

finish() {
  local status=$?
  trap - EXIT
  summary
  if [[ $interrupted -eq 1 ]]; then
    exit 130
  fi
  if [[ ${#bad[@]} -gt 0 ]]; then
    exit 1
  fi
  exit "$status"
}

trap 'interrupted=1' INT
trap finish EXIT

step() {
  local body first rc
  step_n=$((step_n + 1))
  if [[ $# -eq 0 ]]; then
    echo "========== [$step_n] ==========" >&2
    echo "step 缺少命令" >&2
    bad+=("[$step_n] 缺少命令")
    return
  fi
  body=$1
  first=$(printf '%s\n' "$body" | sed -n '/[^[:space:]]/{p;q;}')

  echo "========== [$step_n] $(date '+%F %T') ==========" >&2
  printf '%s\n' "$body" >&2
  # 不用 || 接在子 shell 后面，否则子 shell 里的 set -e 不生效。
  (
    set -euo pipefail
    eval "$body"
  )
  rc=$?
  echo "[$step_n] exit=$rc" >&2
  echo >&2

  if [[ $interrupted -eq 1 ]]; then
    bad+=("[$step_n] $first (interrupted)")
    exit 130
  fi
  if [[ $rc -eq 0 ]]; then
    ok+=("[$step_n] $first")
  else
    bad+=("[$step_n] $first (exit $rc)")
  fi
}

echo "仓库根: $PWD" >&2
echo >&2

# ========== 排队从这里往下改 ==========

# step '
# uv run python scripts/train.py fit \
#   --config configs/experiments/image_classifiter_from_detect_box_head_eye_open_mobilenet_v2_v002.yaml
# '

# step '
# cd exports/sigmastar/mobilenet_v2
# python verify/single_image_similarity_dump_onnx_npz.py \
#   --model onnx/eye_v014.onnx \
#   --image dog.jpg \
#   --mean 0 \
#   --std 255
# '
