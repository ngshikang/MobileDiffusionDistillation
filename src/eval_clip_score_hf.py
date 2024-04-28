# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/mlfoundations/open_clip/tree/37b729bc69068daa7e860fb7dbcf1ef1d03a4185#usage
# ------------------------------------------------------------------------------------

import os
import argparse
import torch
import open_clip
from PIL import Image
from utils.misc import get_file_list_from_csv
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_txt", type=str, default="./results/bk-sdm-small/im256_clip.txt")
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")   
    parser.add_argument("--img_dir", type=str, default="./results/bk-sdm-small/im256")  
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def calculate_clip_score(images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    file_list = get_file_list_from_csv(args.data_list)
    score_arr = []
    for i, file_info in tqdm(enumerate(file_list)):
        img_path = os.path.join(args.img_dir, file_info[0])
        val_prompt = file_info[1]           
        image = np.asarray(Image.open(img_path))
        print(image.shape)
        sd_clip_score = calculate_clip_score(image, val_prompt)
        score_arr.append(sd_clip_score)
        
    final_score = sum(score_arr) / len(score_arr)
    
    print(f"FINAL clip-hf score {final_score}")
    print(f"-- sum score {sum(score_arr)}")
    print(f"-- len {len(score_arr)}")
    
    with open(args.save_txt, 'w') as f:
        f.write(f"FINAL clip-hf score {final_score}\n")
        f.write(f"-- sum score {sum(score_arr)}\n")
        f.write(f"-- len {len(score_arr)}\n")