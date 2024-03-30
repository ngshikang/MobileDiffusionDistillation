#!/bin/bash

source activate bk-sdm

StartTime=$(date +%s)

python ./src/generate_single_prompt_normal.py \
--val_prompt "a boy at a playground" \
--num_inference_steps 5 \
--device cpu \
--save_dir ./inference_results/ --unet_path ./results/nscc_ckpt_20000_absreality_distill 
# --unet_path ./results/nscc_ckpt_15000_dream8_distill \

EndTime=$(date +%s)
echo "** Loading the model and inferencing takes $(($EndTime - $StartTime)) seconds."

