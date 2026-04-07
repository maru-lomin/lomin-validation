#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트(데이터·result 경로 기준). 기본: 이 스크립트가 있는 디렉터리
# SCRIPT_DIR=$(pwd)
HOST_ROOT=$(pwd)
IMAGE="evaluator:latest"

# gt.json / images / images_kv 공통 상위 디렉터리 (visualize.py 기본은 dataset_sampled)
# 예: dataset_sampled_10 을 쓰려면  DATASET_DIR=dataset_sampled_10 ./docker_run.sh
DATASET_DIR=./dataset_sampled_10

# uv_run.sh와 동일: 호스트의 dataset/result를 /data에 마운트하고,
# 앱·가상환경은 이미지의 /app을 사용 (uv run --directory /app).
# 마운트된 /data(호스트 프로젝트 루트)에 쓰려면 호스트 UID/GID와 맞추는 것이 안전합니다.
# NFS root_squash 등으로 컨테이너 root가 쓰기 거부되는 경우가 많습니다.
run_in_container() {
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e MPLCONFIGDIR=/tmp/mplconfig \
    -e UV_CACHE_DIR=/tmp/uv-cache \
    -v "${HOST_ROOT}:/data" \
    -w /data \
    "${IMAGE}" \
    uv run --directory /app "$@"
}

## 시각화 (GT VIA JSON: ${DATASET_DIR}/gt.json)
# run_in_container visualize.py --via-json "/data/${DATASET_DIR}/gt.json" \
#   --images-dir "/data/${DATASET_DIR}/images" --output-dir "/data/${DATASET_DIR}/images_kv"

## API 요청
run_in_container request_api.py --img-dir "/data/${DATASET_DIR}/images" \
  --result-dir /data/result

## 평가
run_in_container evaluate.py --gt-json "/data/${DATASET_DIR}/gt.json" --result-json /data/result/result.json \
  --detail-txt /data/result/detail.txt --detail-xlsx /data/result/detail.xlsx --summary-txt /data/result/summary.txt
