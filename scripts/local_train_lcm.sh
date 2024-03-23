#!/bin/bash

source activate bk-sdm

MODEL_NAME="lykon/dreamshaper-7"
ADAPTER_ID="latent-consistency/lcm-lora-sdv1-5"
TRAIN_DATA_DIR="./data/laion_aes/preprocessed_212k" # please adjust it if needed
UNET_CONFIG_PATH="./src/unet_config"

UNET_NAME="bk_tiny" # option: ["bk_base", "bk_small", "bk_tiny"]
OUTPUT_DIR="./results/kd_"$UNET_NAME # please adjust it if needed

BATCH_SIZE=16
GRAD_ACCUMULATION=4

StartTime=$(date +%s)

CUDA_VISIBLE_DEVICES=0 accelerate launch ./src/lcm_custom_distillation.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $TRAIN_DATA_DIR\
  --use_ema \
  --resolution 512 --center_crop --random_flip \
  --train_batch_size $BATCH_SIZE \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate 5e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --report_to="wandb" \
  --max_train_steps=500000 \
  --seed 1234 \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --checkpointing_steps 1000 \
  --valid_steps 100 \
  --lambda_sd 1.0 --lambda_kd_output 1.0 --lambda_kd_feat 1.0 \
  --use_copy_weight_from_teacher \
  --unet_config_path $UNET_CONFIG_PATH --unet_config_name $UNET_NAME \
  --output_dir $OUTPUT_DIR \
  --valid_prompt "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors" \
  --adapter_id $ADAPTER_ID

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."

