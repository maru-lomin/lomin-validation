#!/usr/bin/env bash
set -euo pipefail

#BASE_DIR=v7-정량평가-v1
BASE_DIR=v13_855

# TP=4(GPU 2-5) eval-llm 기준: 동시 요청을 올려 continuous batching 활용.
# 필요 시 MAX_WORKERS=16 ./eval_llm.sh 처럼 덮어쓸 수 있음.
MAX_WORKERS="${MAX_WORKERS:-64}"

uv run eval_llm.py \
  --pred_gt_file ./${BASE_DIR}/detail.xlsx \
  --eval_prompt_file ./eval_prompt.txt \
  --output_file ./${BASE_DIR}/eval_llm.csv \
  --vllm_url http://localhost:8002 \
  --model auto \
  --max-workers "${MAX_WORKERS}" \
  --no-echo-llm

uv run eval_llm.py --summary_accuracy \
  --result_csv ./${BASE_DIR}/eval_llm.csv \
  --threshold 8
