#!/usr/bin/env bash
set -u

CONFIG_DIR="tmp/test_configs"

success_list=()
fail_list=()

for cfg in "$CONFIG_DIR"/*.yaml; do
  echo "Running config: $cfg"
  if python scripts/train.py fit \
    --trainer.fast_dev_run true \
    --config "$cfg" ; then       # --trainer.fast_dev_run true
    echo "Command succeeded for config: $cfg"
    success_list+=("$cfg")
  else
    echo "Command failed for config: $cfg, skipping..."
    fail_list+=("$cfg")
  fi
done

echo
echo "========== Batch Summary =========="
echo "Success (${#success_list[@]}):"
for cfg in "${success_list[@]}"; do
  echo "  - $cfg"
done

echo
echo "Failed (${#fail_list[@]}):"
for cfg in "${fail_list[@]}"; do
  echo "  - $cfg"
done