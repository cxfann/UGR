

DATASET=Office
DATA_PATH=./data/processed_data
RESULTS_DIR="./results/$DATASET"

PORT=$((RANDOM % 10000 + 10000))

torchrun --nproc_per_node=8 --master_port=$PORT ./eval/test.py \
    --ckpt_path YOUR_SFT_CKPT_PATH \
    --ckpt_path2 YOUR_RL_CKPT_PATH \
    --base_model YOUR_BASE_MODEL_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file ${RESULTS_DIR}/results.json \
    --test_batch_size 1 \
    --num_beams 10 \
    --test_prompt_ids 0 \
    --metrics hit@1,hit@3,hit@5,hit@10,ndcg@3,ndcg@5,ndcg@10 \
    --index_file .index.json
