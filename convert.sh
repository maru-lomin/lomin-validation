# BASE_DIR="./v7-정량평가-v1"
BASE_DIR="./v13_inspect_mask_1143"
IMAGE_DIR="../inference-pipeline/평가-100/v1"
# IMAGE_DIR="${BASE_DIR}/images"
# IMAGE_DIR="./rq_sampled_100/images"
# IMAGE_DIR="./정성평가/images"

python3 demo_to_gt.py \
  --demo-result-dir "${BASE_DIR}/result" \
  --dataset-dir "${IMAGE_DIR}" \
  --output "${BASE_DIR}/result.json"

#python3 excel_to_gt.py \
#  --excel rq_sampled_100/answer_sheet.xlsx \
#  --dataset-dir ${BASE_DIR}/images \
#  --output ${BASE_DIR}/gt.json

#python3 excel_to_gt.py \
#  --excel "./원본 2,000건/answer_sheet_modified_v11.xlsx" \
#  --dataset-dir "../inference-pipeline/평가-100/v1" \
#  --output "${BASE_DIR}/gt.json"
