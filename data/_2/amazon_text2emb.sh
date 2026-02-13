cd ./data/_2

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 amazon_text2emb.py \
    --dataset Office \
    --root YOUR_ROOT_PATH \
    --plm_checkpoint YOUR_PLM_CHECKPOINT