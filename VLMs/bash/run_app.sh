#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
export PYTHONPATH=/home/jovyan/shared/tienhuu060102/data-petct/shared_codes/ViReportGen/VLMs:/home/jovyan/shared/tienhuu060102/data-petct/shared_codes/ViReportGen/VLMs/llava/model/multimodal_encoder:$PYTHONPATH

PROMPT_VERSION=v1
MODEL_VERSION=llava-med-v1.5-mistral-7b

################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
# PROMPT_VERSION=plain
################## LLaMA-2 ##################

# --data_path /home/user01/aiotlab/thaind/data_desc_conv_train.json \
# --eval_data_path /home/user01/aiotlab/thaind/data_desc_conv_test.json \
    # --pretrain_mm_mlp_adapter /home/user01/aiotlab/thaind/LLaVA/checkpoints/ctvit_llavamed-llava-med-v1.5-mistral-7b-pretrain-1epochs/mm_projector.bin \
deepspeed --master_port=29504 llava/infer_ui/be/main.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/MultimodalFM/llava-med-v1.5-mistral-7b \
    --version $PROMPT_VERSION \
    --type PET/CT \
    --image_folder /home/jovyan/shared/tienhuu060102/data-petct/PET_report_paired_fixed \
    --vision_tower /home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/petct_emb/ctvit.89000.pt \
    --pretrain_mm_mlp_adapter /home/jovyan/shared/tienhuu060102/data-petct/shared_codes/ViReportGen/VLMs/checkpoints/ctvit_llavamed_lora/checkpoint-2778/mm_projector.bin \
    --lora_path /home/jovyan/shared/tienhuu060102/data-petct/shared_codes/ViReportGen/VLMs/checkpoints/ctvit_llavamed_lora/checkpoint-2778 \
    --question_file /home/jovyan/shared/tienhuu060102/data-petct/PET_report_paired_fixed/pretrain_data/single_turn/align_train.json \
    --temperature 0.5 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./infer/ctvit_llavamed/train \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb

