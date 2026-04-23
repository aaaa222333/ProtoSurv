#!/bin/bash

# 定义日志基础路径
OUT_BASE="/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/lung1/sim"
LOG_BASE="/home/user/prognosis_lst/evaluate_model/ConvNeXt/logs/lung1/sim"
mkdir -p "$LOG_BASE"

echo "🚀 实验启动中..."


CUDA_VISIBLE_DEVICES=0 python main_sim.py \
    --train_path='/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/train/final_complete_with_paths_labels.csv' \
    --val_path='/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/test_val/final_complete_with_paths_labels.csv' \
    --log_dir="$LOG_BASE" \
    --output_dir="$OUT_BASE" \
    > "$LOG_BASE/train_log_aug.txt" 2>&1 &

PID_AUG=$! # 获取上一个后台命令(Exp 1)的进程号


echo "------------------------------------------------"

echo "PID: $PID_AUG"


wait

echo "🎉 所有训练任务已完成！"