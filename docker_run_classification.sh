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
WORKFLOW_API_KEY=c313d53e2f639dced276f5607dc5013b433aef505e79b63
AUTH_BASE_URL=${BASE_URL}/cognition-api/api/v2/account/login
AUTH_EMAIL=team.data@lomin.ai
AUTH_PASSWORD=1q2w3e4r!!

# --- 데이터셋 (gt.json / images 공통 상위) ---
DATASET_DIR="${DATASET_DIR:-./dataset_classification}"
DATASET_REL="${DATASET_DIR#./}"
DATASET_IMAGES="${APP}/${DATASET_REL}/images"
DATASET_GT="${APP}/${DATASET_REL}/gt.csv"
RESULT_RESPONSE_JSONL="${RESULT_DIR}/response.jsonl"
RESULT_EVALUATION_CSV="${RESULT_DIR}/evaluation.csv"
RESULT_CONFUSION_MATRIX_PNG="${RESULT_DIR}/confusion_matrix.png"

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

run_in_container request_api.py \
  --workflow-url "${WORKFLOW_URL}" \
  --api-key "${WORKFLOW_API_KEY}" \
  --auth-base-url "${AUTH_BASE_URL}" \
  --img-dir "${DATASET_IMAGES}" \
  --result-dir "${RESULT_DIR}"

run_in_container evaluate.py \
  --mode classification \
  --gt-csv "${DATASET_GT}" \
  --response-jsonl "${RESULT_RESPONSE_JSONL}" \
  --evaluation-csv "${RESULT_EVALUATION_CSV}" \
  --confusion-matrix-png "${RESULT_CONFUSION_MATRIX_PNG}"
