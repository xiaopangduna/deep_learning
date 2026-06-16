#!/usr/bin/env bash
set -euo pipefail

echo "==> NPU conversion environment: ubuntu-npu:v2.0.10.2"
python3 --version
pip3 show acuity | awk '/^Version:/{print "acuity " $2}'

AI_SDK_MODELS="${PWD}/convert/allwinner/ai-sdk/models"
if [[ -f "${AI_SDK_MODELS}/env.sh" ]]; then
    echo "==> source env.sh v3"
    pushd "${AI_SDK_MODELS}" >/dev/null
    # shellcheck disable=SC1091
    source env.sh v3
    popd >/dev/null
    echo "    VSIMULATOR_CONFIG=${VSIMULATOR_CONFIG:-}"
else
    echo "WARN: env.sh not found at ${AI_SDK_MODELS}"
fi

if [[ -x "${ACUITY_TOOLKIT_ROOT}/pegasus.py" ]]; then
    python3 "${ACUITY_TOOLKIT_ROOT}/pegasus.py" help 2>/dev/null | head -8
else
    echo "WARN: pegasus.py not found at ${ACUITY_TOOLKIT_ROOT}"
fi

echo "==> workspace: ${PWD}"
echo "    artifacts: ${PWD}/artifacts (create on demand)"
