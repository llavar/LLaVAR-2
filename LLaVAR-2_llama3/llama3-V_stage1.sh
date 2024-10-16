#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --cache_dir /data/shijie/llama3/Meta-Llama-3-8B-Instruct \
    --version plain \
    --data_path /scr/shijie/llava_dpo1/data_generation/final_merged/train/detailed_description_QA_train.json \
    --image_folder /scr/shijie/llava_dpo/training_img \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /data/shijie/llava_llama3/checkpoints/llava-v1.5-8b-pretrain1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
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
    --report_to none \
    --is_multipath_encoder True \
    --input_image_size 384


