#!/bin/bash

# 基础路径
BASE_OUT="/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/huaxi/pull_strength/pull_strength"
BASE_LOG="/home/user/prognosis_lst/evaluate_model/ConvNeXt/logs/huaxi/pull_strength/pull_strength"

# 存储所有进程 PID 的数组
PIDS=()

echo "🚀 开始并发启动多强度实验..."

# 循环：0.2, 0.7 (步长 0.5)
# 如果你想跑 0.2, 0.3, 0.4...0.8，请将 0.5 改为 0.1
for STRENGTH in $(seq 0 0.05 0.8)
do
    # 动态生成带后缀的路径
    CURRENT_OUT="${BASE_OUT}_pull_${STRENGTH}"
    CURRENT_LOG="${BASE_LOG}_pull_${STRENGTH}"

    # 创建文件夹
    mkdir -p "$CURRENT_OUT"
    mkdir -p "$CURRENT_LOG"

    echo "------------------------------------------------"
    echo "🔹 启动后台任务: pull_strength = $STRENGTH"
    echo "📂 日志: $CURRENT_LOG/train_log_aug.txt"

    # 执行训练（注意最后的 & 符号，表示放入后台运行）
    # 如果显存足够，可以都在 0 号卡跑；如果有多张卡，可以手动修改 CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --train_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/train/final_complete_with_paths_labels.csv" \
        --val_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/val/final_complete_with_paths_labels.csv" \
        --log_dir="$CURRENT_LOG" \
        --output_dir="$CURRENT_OUT" \
        --pre_path="/home/user/prognosis_lst/evaluate_model/feature_similiary/huaxi/inference_distance_samples_filtered.csv" \
        --pull_strength="$STRENGTH" \
        > "$CURRENT_LOG/train_log_aug.txt" 2>&1 &

    # 记录该任务的 PID
    PIDS+=($!)
done

echo "------------------------------------------------"
echo "⏳ 所有任务已进入后台，正在等待运行结束..."
echo "当前监控的 PIDs: ${PIDS[@]}"

# 统一等待所有后台任务
wait

echo "🎉 所有不同强度的训练任务已全部完成！"