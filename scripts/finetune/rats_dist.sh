#!/usr/bin/env bash

DATA_PATH='/tsukimi/datasets/Chiba/baseline/datalist/'
OUTPUT_DIR='/tsukimi/datasets/Chiba/baseline/checkpoints'
MODEL_PATH='/tsukimi/datasets/Chiba/baseline/checkpoints/pretrained_weights/vit_b_k710_dl_from_giant.pth'
# MODEL_PATH='/tsukimi/datasets/Chiba/baseline/checkpoints/checkpoint-39/mp_rank_00_model_states.pt'
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 12320 --nnodes=1  --node_rank=0 --master_addr="localhost" \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 2 \
        --input_size 224 \
        --nb_classes 7 \
        --data_set rats \
        --num_sample 2 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --layer_decay 0.75 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --epochs 90 \
        --dist_eval \


# freeze parameters
# learning rate too low
# check dataset, data loader, dump dataset
