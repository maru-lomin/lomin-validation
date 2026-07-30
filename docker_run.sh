#!/usr/bin/env bash
set -euo pipefail

# --- 호스트·이미지 ---
HOST_ROOT=$(pwd)
# IMAGE="${IMAGE:-evaluator:latest}"
IMAGE=evaluator:latest
APP="/app"
RESULT_DIR="${APP}/result"
PYTHON_BIN="/install/.venv/bin/python"

BASE_URL=https://beta.zixy.io
WORKFLOW_URL=${BASE_URL}/cognition-api/api/v1/workflows/api/apis
WORKFLOW_API_KEY=03e57dac1847ddfa296b8813f17c21c21de945e330f39b7
AUTH_BASE_URL=${BASE_URL}/cognition-api/api/v2/account/login
AUTH_EMAIL=team.data@lomin.ai
AUTH_PASSWORD=1q2w3e4r!!

# --- 데이터셋 (gt.json / images 공통 상위) ---
# DATASET_IMAGES="./images"
# DATASET_GT="./convert_sample/gt.json"
# DATASET_GT="./한도_공제/gt.json"

run_in_container() {
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e MPLCONFIGDIR=/tmp/mplconfig \
    -e UV_CACHE_DIR=/tmp/uv-cache \
    -v "${HOST_ROOT}:${APP}" \
    -w "${APP}" \
    "${IMAGE}" \
    "${PYTHON_BIN}" "$@"
}

## 시각화 (GT VIA JSON: ${DATASET_DIR}/gt.json)
# --via-json "${DATASET_GT}" \
# run_in_container visualize.py \
#   --via-json "${RESULT_DIR}/result.json" \
#   --images-dir "${DATASET_IMAGES}" \
#   --output-dir "${RESULT_DIR}/images_kv_result"
#   # --output-dir "${APP}/${DATASET_REL}/images_kv"

# run_in_container request_api.py \
#   --workflow-url "${WORKFLOW_URL}" \
#   --api-key "${WORKFLOW_API_KEY}" \
#   --auth-base-url "${AUTH_BASE_URL}" \
#   --img-dir "${DATASET_IMAGES}" \
#   --result-dir "${RESULT_DIR}"

#BASE_DIR="./v7-정량평가-v1"
BASE_DIR="./v13_855"
run_in_container evaluate.py \
  --mode kv \
  --kv-scoring edit_distance \
  --gt-json "${BASE_DIR}/gt.json" \
  --result-json "${BASE_DIR}/result.json" \
  --detail-txt "${BASE_DIR}/detail.txt" \
  --detail-xlsx "${BASE_DIR}/detail.xlsx" \
  --summary-txt "${BASE_DIR}/summary.txt"
  # --kv-scoring edit_distance \
  # --kv-scoring char_multiset \

run_in_container generate_report.py \
  --summary-txt "${BASE_DIR}/summary.txt" \
  --detail-xlsx "${BASE_DIR}/detail.xlsx" \
  --output-xlsx "${BASE_DIR}/report.xlsx"
