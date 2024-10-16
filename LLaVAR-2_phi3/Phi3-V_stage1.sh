#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --cache_dir /data/shijie/llava_phi3/microsoft_Phi-3-mini-4k-instruct \
    --version plain \
    --data_path /home/shijie/llava_dpo/data_generation/single_round.ndjson \
    --image_folder /home/shijie/llava_dpo/data_generation/training_img \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /data/shijie/llava_phi3/checkpoints/llava-v1.5-phi3-mini-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --is_multipath_encoder True \
    --input_image_size 384