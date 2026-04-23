#!/bin/bash
# eval "$(conda shell.bash hook)"
# conda activate clip2


accelerate launch \
  --num_processes 1 \
  --gpu_ids 1 \
  feature_extract.py \
  --testpath="/home/user/prognosis_lst/data/pretrain/val_cleaned.csv" \
  --tepath_name="image" \
  --telabel_name="text"