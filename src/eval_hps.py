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
import hpsv2
from tqdm import tqdm

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
    
    file_list = get_file_list_from_csv(args.data_list)
    results = []

    for i, file_info in tqdm(enumerate(file_list)):
        img_path = os.path.join(args.img_dir, file_info[0])
        val_prompt = file_info[1]                   
        result = hpsv2.score(img_path, val_prompt, hps_version="v2.1", device=args.device) 
        print(result)
        
    final_score = sum(result) / len(result)
    with open(args.save_txt, 'w') as f:
        f.write(f"FINAL hps score {final_score}\n")
        f.write(f"-- sum score {sum(result)}\n")
        f.write(f"-- len {len(result)}\n")
    
