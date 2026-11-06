#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE="${1:-npu}"
[[ $# -le 1 && ( "$MODE" == npu || "$MODE" == cpu ) ]] || { echo 'Usage: bash run.sh [npu|cpu]'; exit 2; }
: "${ASCEND_HOME_PATH:?Source your CANN set_env.sh first}"
PYTHON="${PYTHON:-python3}"
BUILD_JOBS="${BUILD_JOBS:-4}"
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'BUILD_JOBS must be positive'; exit 2; }
BUILD="build-$MODE"
echo "[1/4] 正在编译算子（${MODE} 模式）..."
cmake -S . -B "$BUILD" -DCMAKE_ASC_RUN_MODE="$MODE" -DCMAKE_ASC_ARCHITECTURES="${NPU_ARCH:-dav-2201}"
cmake --build "$BUILD" --parallel "$BUILD_JOBS"
echo "[2/4] 正在生成样例测试数据..."
"$PYTHON" sample.py generate --directory "$BUILD"
echo "[3/4] 正在运行算子..."
(cd "$BUILD" && ./sample "${DEVICE_ID:-0}")
echo "[4/4] 正在校验输出数据..."
"$PYTHON" sample.py verify --directory "$BUILD"
