cd ./data/_3

CUDA_VISIBLE_DEVICES=0 python rqvae.py \
      --data_path YOUR_NPY_PATH \
      --ckpt_dir ./output/Office \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 1024 \
      --weight_decay 1e-4 \
      --lr_scheduler_type linear \
      --num_emb_list 256 256 256 256 \
      --sk_epsilons 0.0 0.0 0.0 0.003 \
      --layers 2048 1024 512 256 128 64 