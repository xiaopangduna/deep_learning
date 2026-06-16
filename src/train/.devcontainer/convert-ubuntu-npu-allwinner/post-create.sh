#!/usr/bin/env bash
set -euo pipefail

echo "==> NPU conversion environment: ubuntu-npu:v2.0.10.2"
python3 --version
pip3 show acuity | awk '/^Version:/{print "acuity " $2}'

if [[ -x "${ACUITY_TOOLKIT_ROOT}/pegasus.py" ]]; then
    python3 "${ACUITY_TOOLKIT_ROOT}/pegasus.py" help 2>/dev/null | head -8
else
    echo "WARN: pegasus.py not found at ${ACUITY_TOOLKIT_ROOT}"
fi

echo "==> workspace: ${PWD}"
echo "    artifacts: ${PWD}/artifacts (create on demand)"
