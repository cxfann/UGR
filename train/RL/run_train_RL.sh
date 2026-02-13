
DATASET=Office
DATA_PATH=./data/processed_data

PORT=$((RANDOM % 10000 + 10000))

torchrun --nproc_per_node=8 --master_port=$PORT  ./train/RL/train.py \
    --base_model YOUR_BASE_MODEL_PATH \
    --ckpt_path YOUR_SFT_CKPT_PATH \
    --output_dir YOUR_OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_generations 16 \
    --learning_rate 2e-6 \
    --epochs 2 \
    --index_file .index.json \
    --lora_r 16 \
    --lora_alpha 32 \
    --deepspeed ./train/SFT/config/ds_z2_bf16.json \
    --beam_search \
    --test_during_training \
    --alpha_min 0.0 \
    --coeff_partial 0.0 \
    --coeff_wrong 0.5 \
    --smart_initialize_conf_tokens