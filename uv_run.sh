## 시각화
uv run visualize.py --via-json result/result.json --images-dir dataset_sampled/images --output-dir dataset_sampled/images_kv

## API 요청
uv run request_api.py --img-dir dataset_sampled_10/images --result-dir result

## 평가
uv run evaluate.py --gt-json dataset_sampled_10/gt.json --result-json result/result.json \
    --detail-txt result/detail.txt --detail-xlsx result/detail.xlsx --summary-txt result/summary.txt