#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat
export PYTHONPATH=/home/jovyan/shared/tienhuu060102/data-petct/shared_codes/ViReportGen/VLMs:$PYTHONPATH
# MODEL_VERSION=llava-med-v1.5-mistral-7b

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=v1
########### DO NOT CHANGE ###########

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/MultimodalFM/llava-med-v1.5-mistral-7b \
    --version $PROMPT_VERSION \
    --type PET/CT \
    --data_path /home/jovyan/shared/tienhuu060102/data-petct/PET_report_paired_fixed/pretrain_data/single_turn/align_train.json \
    --eval_data_path /home/jovyan/shared/tienhuu060102/data-petct/PET_report_paired_fixed/pretrain_data/single_turn/align_val.json \
    --image_folder /home/jovyan/shared/tienhuu060102/data-petct/PET_report_paired_fixed \
    --vision_tower /home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/petct_emb/ctvit.89000.pt \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/jovyan/shared/tienhuu060102/data-petct/shared_codes/ViReportGen/VLMs/checkpoints/align_perceiver/ctvit_llavamed \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb

