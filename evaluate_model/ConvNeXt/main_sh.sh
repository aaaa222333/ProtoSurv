 #!/bin/bash
 # eval "$(conda shell.bash hook)"
 # conda activate clip2

#
# python main_origin.py --train_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/train/final_complete_with_paths_labels.csv" --val_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/val/final_complete_with_paths_labels.csv" --output_dir="/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/huaxi/Resnet" --log_dir="/home/user/prognosis_lst/evaluate_model/ConvNeXt/logs/huaxi/Resnet"
#
# python main.py --train_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/train/final_complete_with_paths_labels.csv" --val_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/val/final_complete_with_paths_labels.csv" --output_dir="/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/huaxi/Resnet" --log_dir="/home/user/prognosis_lst/evaluate_model/ConvNeXt/logs/huaxi/Resnet" --pre_path="/home/user/prognosis_lst/evaluate_model/feature_similiary/huaxi/inference_distance_samples_filtered.csv"
#
#
# python main.py --train_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/train/final_complete_with_paths_labels.csv" --val_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/test_val/final_complete_with_paths_labels.csv" --output_dir="/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/lung1/Resnet" --log_dir="/home/user/prognosis_lst/evaluate_model/ConvNeXt/logs/lung1/Resnet" --pre_path="/home/user/prognosis_lst/evaluate_model/feature_similiary/huaxi/inference_distance_samples_filtered.csv"
#
#
#!/bin/bash
#
# 定义日志基础路径
#OUT_BASE="/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/lung1/normal"
#LOG_BASE="/home/user/prognosis_lst/evaluate_model/ConvNeXt/logs/lung1/normal"
#mkdir -p "$LOG_BASE"
#
#echo "🚀 实验启动中..."
#
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --train_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/train/final_complete_with_paths_labels.csv" \
#    --val_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/test_val/final_complete_with_paths_labels.csv" \
#    --log_dir="$LOG_BASE" \
#    --output_dir="$OUT_BASE" \
#    --pre_path="/home/user/prognosis_lst/evaluate_model/feature_similiary/lung1/inference_distance_samples_filtered.csv" \
#    > "$LOG_BASE/train_log_aug.txt" 2>&1 &
#
#PID_AUG=$! # 获取上一个后台命令(Exp 1)的进程号
#
#
#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --train_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/train/final_complete_with_paths_labels.csv" \
#    --val_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/test_val/final_complete_with_paths_labels.csv" \
#    --log_dir="$LOG_BASE" \
#    --output_dir="$OUT_BASE" \
#    --pre_path="" \
#    > "$LOG_BASE/train_log_base.txt" 2>&1 &
#
#PID_BASE=$! # 获取上一个后台命令(Exp 1)的进程号
#
#
#
#echo "------------------------------------------------"
#
#echo "🔹 扩增组 (Augmentation) PID: $PID_AUG"
#echo "🔹 对照组 (Baseline)     PID: $PID_BASE"
#
#wait
#
#echo "🎉 所有训练任务已完成！"
##

OUT_BASE="/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/huaxi/normal"
LOG_BASE="/home/user/prognosis_lst/evaluate_model/ConvNeXt/logs/huaxi/normal"
mkdir -p "$LOG_BASE"

echo "🚀 实验启动中..."

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/train/final_complete_with_paths_labels.csv" \
    --val_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/val/final_complete_with_paths_labels.csv" \
    --log_dir="$LOG_BASE" \
    --output_dir="$OUT_BASE" \
    --pre_path="/home/user/prognosis_lst/evaluate_model/feature_similiary/huaxi/inference_distance_samples_filtered.csv" \
    > "$LOG_BASE/train_log_aug.txt" 2>&1 &

PID_AUG=$! # 获取上一个后台命令(Exp 1)的进程号


CUDA_VISIBLE_DEVICES=0 python main.py \
    --train_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/train/final_complete_with_paths_labels.csv" \
    --val_path="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/val/final_complete_with_paths_labels.csv" \
    --log_dir="$LOG_BASE" \
    --output_dir="$OUT_BASE" \
    --pre_path="" \
    > "$LOG_BASE/train_log_base.txt" 2>&1 &

PID_BASE=$! # 获取上一个后台命令(Exp 1)的进程号



echo "------------------------------------------------"

echo "🔹 扩增组 (Augmentation) PID: $PID_AUG"
echo "🔹 对照组 (Baseline)     PID: $PID_BASE"

wait

echo "🎉 所有训练任务已完成！"