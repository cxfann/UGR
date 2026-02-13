cd ./data/_1

python amazon18_process.py \
    --input_path YOUR_INPUT_PATH \
    --dataset Office \
    --user_k 5 \
    --st_year 2016 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 11 \
    --output_path ../processed_data
