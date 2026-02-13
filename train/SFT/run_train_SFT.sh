
DATASET=Office
DATA_PATH=./data/processed_data

PORT=$((RANDOM % 10000 + 10000))

torchrun --nproc_per_node=8 --master_port=$PORT  ./train/SFT/lora_finetune.py \
    --base_model YOUR_BASE_MODEL_PATH \
    --output_dir YOUR_OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --learning_rate 2e-4 \
    --epochs 10 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index.json \
    --lora_r 16 \
    --lora_alpha 32 \
    --only_train_response \
    --save_and_eval_strategy steps \
    --save_and_eval_steps 0.05
