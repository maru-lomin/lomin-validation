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
DATASET_DIR="${DATASET_DIR:-./dataset_sampled_10_viatool}"
DATASET_REL="${DATASET_DIR#./}"
DATASET_IMAGES="${APP}/${DATASET_REL}/images"
DATASET_GT="${APP}/${DATASET_REL}/gt.json"

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
# run_in_container visualize.py \
#   --via-json "${DATASET_GT}" \
#   --images-dir "${DATASET_IMAGES}" \
#   --output-dir "${APP}/${DATASET_REL}/images_kv"

run_in_container request_api.py \
  --workflow-url "${WORKFLOW_URL}" \
  --api-key "${WORKFLOW_API_KEY}" \
  --auth-base-url "${AUTH_BASE_URL}" \
  --img-dir "${DATASET_IMAGES}" \
  --result-dir "${RESULT_DIR}"

run_in_container evaluate.py \
  --gt-json "${DATASET_GT}" \
  --result-json "${RESULT_DIR}/result.json" \
  --detail-txt "${RESULT_DIR}/detail.txt" \
  --detail-xlsx "${RESULT_DIR}/detail.xlsx" \
  --summary-txt "${RESULT_DIR}/summary.txt"
