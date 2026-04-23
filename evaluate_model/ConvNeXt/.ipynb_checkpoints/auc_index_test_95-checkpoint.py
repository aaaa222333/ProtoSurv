import argparse
import datetime
import time
import torch
from engine import test
from datasets_test import prepare_dataset
from models import Resnet_1D
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from pycox.evaluation import EvalSurv

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation Script', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--finetune', default='/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/lung1/Resnet/pre/checkpoint-best.pth', help='权重路径')
    parser.add_argument('--test_path', type=str, default="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/test_val/final_complete_with_paths_labels.csv")
    parser.add_argument('--column_names', type=str, default="feature_path")
    parser.add_argument('--labels_name', type=str, default="label")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_mem", type=bool, default=True)
    
    return parser
def calculate_c_index_with_ci(df, target_time=1.0, n_iterations=1000):
    y_true = df['label'].values
    y_surv_prob = (1 - df['pos_probs'].values)
    durations_test = np.full(len(y_true), target_time)

    # A. 计算全量数据集上的结果 (作为 Mean)
    surv_full = pd.DataFrame(y_surv_prob.reshape(1, -1), index=[target_time])
    ev_full = EvalSurv(surv_full, durations_test, y_true.astype(np.int64), censor_surv='km')
    full_score = ev_full.concordance_td()

    # B. Bootstrap 仅用于计算 CI
    stats = []
    print(f"正在通过 Bootstrap 计算 C-index 的置信区间...")
    for i in range(n_iterations):
        indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)
        sub_y_true = y_true[indices].astype(np.int64)
        if len(np.unique(sub_y_true)) < 2: continue
        
        sub_y_prob = y_surv_prob[indices]
        sub_durations = durations_test[indices]
        
        surv_df = pd.DataFrame(sub_y_prob.reshape(1, -1), index=[target_time])
        ev = EvalSurv(surv_df, sub_durations, sub_y_true, censor_surv='km')
        stats.append(ev.concordance_td())

    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return full_score, lower, upper

# ---------------------------------------------------------
# 2. 分类指标计算 (均值取自全量数据集)
# ---------------------------------------------------------
def calculate_classification_metrics(df, n_iterations=1000):
    y_true = df['label'].values
    y_scores = df['pos_probs'].values
    y_pred = df['prediction'].values

    # A. 直接计算完整测试集的结果
    raw_results = {
        'auc': roc_auc_score(y_true, y_scores),
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'pre': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }

  
    boot_stats = {'auc': [], 'acc': [], 'f1': [], 'pre': [], 'recall': []}
    print(f"正在通过 Bootstrap 计算分类指标的置信区间...")
    for i in range(n_iterations):
        indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)
        sub_true = y_true[indices]
        if len(np.unique(sub_true)) < 2: continue

        sub_scores = y_scores[indices]
        sub_pred = y_pred[indices]

        boot_stats['auc'].append(roc_auc_score(sub_true, sub_scores))
        boot_stats['acc'].append(accuracy_score(sub_true, sub_pred))
        boot_stats['f1'].append(f1_score(sub_true, sub_pred, zero_division=0))
        boot_stats['pre'].append(precision_score(sub_true, sub_pred, zero_division=0))
        boot_stats['recall'].append(recall_score(sub_true, sub_pred, zero_division=0))

    final_results = {}
    for key, mean_val in raw_results.items():
        lower = np.percentile(boot_stats[key], 2.5)
        upper = np.percentile(boot_stats[key], 97.5)
        final_results[key] = (mean_val, lower, upper)

    return final_results

# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader_test, len_dataset_test = prepare_dataset(args)

    model = Resnet_1D.ResNet1D()

    # 权重加载逻辑 (保持原样)
    if args.finetune:
        ckpt = torch.load(args.finetune, map_location='cpu')
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        print(f"成功加载权重: {args.finetune}")

    model.to(device)
    model.eval()

    print(f"测试样本数量: {len_dataset_test}")
    start_time = time.time()

    # 获取全量测试集推理结果
    df_results = test(data_loader_test, model, device)

    # 分别计算指标
    c_val, c_low, c_high = calculate_c_index_with_ci(df_results)
    clf_metrics = calculate_classification_metrics(df_results)

    # -----------------------------------------------------
    # 输出报表 (这里的 Mean 是整个测试集的真实得分)
    # -----------------------------------------------------
    print("\n" + "="*60)
    print(f"{'指标 (Full Test Set)':<20} | {'结果 (Score)':<10} | {'95% CI (Bootstrap)':<20}")
    print("-" * 60)
    
    print(f"{'C-index (pycox)':<20} | {c_val:.4f}     | [{c_low:.4f} - {c_high:.4f}]")
    
    display_map = {
        'auc': 'AUC (ROC)', 'acc': 'Accuracy', 'f1': 'F1-Score',
        'pre': 'Precision', 'recall': 'Recall'
    }

    for key, name in display_map.items():
        score, low, high = clf_metrics[key]
        print(f"{name:<20} | {score:.4f}     | [{low:.4f} - {high:.4f}]")
        
    print("="*60)
    print(f'评估总耗时: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')

if __name__ == '__main__':
    # 这里引用你原本的 get_args_parser
    parser = argparse.ArgumentParser('Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)