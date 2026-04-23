# import argparse
# import datetime
# import time
# import torch
# from engine import test
# from datasets_test import prepare_dataset
# from models import Resnet_1D,FC
# import numpy as np
# import pandas as pd
# from sklearn.utils import resample
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
# from pycox.evaluation import EvalSurv
#
# def get_args_parser():
#     parser = argparse.ArgumentParser('Evaluation Script', add_help=False)
#     parser.add_argument('--batch_size', default=10, type=int)
#     parser.add_argument('--finetune', default='/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/huaxi/pull_strength/pull_strength_pull_0.75/generate/checkpoint-best.pth', help='权重路径')
#     parser.add_argument('--test_path', type=str, default="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/val/final_complete_with_paths_labels.csv")
#     parser.add_argument('--column_names', type=str, default="feature_path")
#     parser.add_argument('--labels_name', type=str, default="label")
#     parser.add_argument("--num_workers", type=int, default=8)
#     parser.add_argument("--pin_mem", type=bool, default=True)
#
#     return parser
# def calculate_c_index_with_ci(df, target_time=1.0, n_iterations=1000):
#     y_true = df['label'].values
#     y_surv_prob = (1 - df['pos_probs'].values)
#     durations_test = np.full(len(y_true), target_time)
#
#     # A. 计算全量数据集上的结果 (作为 Mean)
#     surv_full = pd.DataFrame(y_surv_prob.reshape(1, -1), index=[target_time])
#     ev_full = EvalSurv(surv_full, durations_test, y_true.astype(np.int64), censor_surv='km')
#     full_score = ev_full.concordance_td()
#
#     # B. Bootstrap 仅用于计算 CI
#     stats = []
#     print(f"正在通过 Bootstrap 计算 C-index 的置信区间...")
#     for i in range(n_iterations):
#         indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)
#         sub_y_true = y_true[indices].astype(np.int64)
#         if len(np.unique(sub_y_true)) < 2: continue
#
#         sub_y_prob = y_surv_prob[indices]
#         sub_durations = durations_test[indices]
#
#         surv_df = pd.DataFrame(sub_y_prob.reshape(1, -1), index=[target_time])
#         ev = EvalSurv(surv_df, sub_durations, sub_y_true, censor_surv='km')
#         stats.append(ev.concordance_td())
#
#     lower = np.percentile(stats, 2.5)
#     upper = np.percentile(stats, 97.5)
#     return full_score, lower, upper
#
# # ---------------------------------------------------------
# # 2. 分类指标计算 (均值取自全量数据集)
# # ---------------------------------------------------------
# def calculate_classification_metrics(df, n_iterations=1000):
#     y_true = df['label'].values
#     y_scores = df['pos_probs'].values
#     y_pred = df['prediction'].values
#
#     # A. 直接计算完整测试集的结果
#     raw_results = {
#         'auc': roc_auc_score(y_true, y_scores),
#         'acc': accuracy_score(y_true, y_pred),
#         'f1': f1_score(y_true, y_pred, zero_division=0),
#         'pre': precision_score(y_true, y_pred, zero_division=0),
#         'recall': recall_score(y_true, y_pred, zero_division=0)
#     }
#
#
#     boot_stats = {'auc': [], 'acc': [], 'f1': [], 'pre': [], 'recall': []}
#     print(f"正在通过 Bootstrap 计算分类指标的置信区间...")
#     for i in range(n_iterations):
#         indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)
#         sub_true = y_true[indices]
#         if len(np.unique(sub_true)) < 2: continue
#
#         sub_scores = y_scores[indices]
#         sub_pred = y_pred[indices]
#
#         boot_stats['auc'].append(roc_auc_score(sub_true, sub_scores))
#         boot_stats['acc'].append(accuracy_score(sub_true, sub_pred))
#         boot_stats['f1'].append(f1_score(sub_true, sub_pred, zero_division=0))
#         boot_stats['pre'].append(precision_score(sub_true, sub_pred, zero_division=0))
#         boot_stats['recall'].append(recall_score(sub_true, sub_pred, zero_division=0))
#
#     final_results = {}
#     for key, mean_val in raw_results.items():
#         lower = np.percentile(boot_stats[key], 2.5)
#         upper = np.percentile(boot_stats[key], 97.5)
#         final_results[key] = (mean_val, lower, upper)
#
#     return final_results
#
# # ---------------------------------------------------------
# # 3. 主程序
# # ---------------------------------------------------------
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_loader_test, len_dataset_test = prepare_dataset(args)
#
#     model = Resnet_1D.ResNet1D()
#     # model = FC.FC()
#
#     # 权重加载逻辑 (保持原样)
#     if args.finetune:
#         ckpt = torch.load(args.finetune, map_location='cpu')
#         state_dict = ckpt["model"] if "model" in ckpt else ckpt
#         new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
#         model.load_state_dict(new_state_dict, strict=True)
#         print(f"成功加载权重: {args.finetune}")
#
#     model.to(device)
#     model.eval()
#
#     print(f"测试样本数量: {len_dataset_test}")
#     start_time = time.time()
#
#     # 获取全量测试集推理结果
#     df_results = test(data_loader_test, model, device)
#
#     # 分别计算指标
#     c_val, c_low, c_high = calculate_c_index_with_ci(df_results)
#     clf_metrics = calculate_classification_metrics(df_results)
#
#     # -----------------------------------------------------
#     # 输出报表 (这里的 Mean 是整个测试集的真实得分)
#     # -----------------------------------------------------
#     print("\n" + "="*60)
#     print(f"{'指标 (Full Test Set)':<20} | {'结果 (Score)':<10} | {'95% CI (Bootstrap)':<20}")
#     print("-" * 60)
#
#     print(f"{'C-index (pycox)':<20} | {c_val:.4f}     [{c_low:.4f} - {c_high:.4f}]")
#
#     display_map = {
#         'auc': 'AUC (ROC)', 'acc': 'Accuracy', 'f1': 'F1-Score',
#         'pre': 'Precision', 'recall': 'Recall'
#     }
#
#
#     for key, name in display_map.items():
#         score, low, high = clf_metrics[key]
#         print(f"{name:<20} | {score:.4f}     [{low:.4f} - {high:.4f}]")
#
#     print("="*60)
#     print(f'评估总耗时: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
#
# if __name__ == '__main__':
#     # 这里引用你原本的 get_args_parser
#     parser = argparse.ArgumentParser('Evaluation Script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     main(args)

import argparse
import datetime
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from models import Resnet_1D
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
from pycox.evaluation import EvalSurv
from tqdm import tqdm


def prepare_dataset(input_csv):
    """使用 datasets 库加载数据"""
    dataset = load_dataset('csv', data_files={"test": input_csv})

    feature_path = "feature_path"
    labels_name = "label"

    def preprocess(examples):
        features = [torch.load(feature, weights_only=True) for feature in examples[feature_path]]
        paths = [path for path in examples[feature_path]]

        normalized_features = []
        for feature in features:
            normalized = (feature - 0.5) / 0.5  # 映射到 [-1, 1]
            normalized_features.append(normalized)

        examples["features"] = normalized_features
        examples["paths"] = paths
        return examples

    test_dataset = dataset["test"].with_transform(preprocess)

    def collate_fn(examples):
        features = torch.stack([example["features"] for example in examples])
        features = features.to(memory_format=torch.contiguous_format).float()
        labels = [example[labels_name] for example in examples]
        paths = [example["paths"] for example in examples]
        return {"features": features, "labels": labels, "paths": paths}

    sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        sampler=sampler_test,
        collate_fn=collate_fn,
        batch_size=10,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    return data_loader_test, len(test_dataset)


def calculate_c_index_with_ci(df, target_time=1.0, n_iterations=1000):
    """计算 C-index 及其置信区间"""
    y_true = df['label'].values
    y_surv_prob = (1 - df['pos_probs'].values)
    durations_test = np.full(len(y_true), target_time)

    # 全量数据集上的结果
    surv_full = pd.DataFrame(y_surv_prob.reshape(1, -1), index=[target_time])
    ev_full = EvalSurv(surv_full, durations_test, y_true.astype(np.int64), censor_surv='km')
    full_score = ev_full.concordance_td()

    # Bootstrap 计算 CI
    stats = []
    print(f"正在通过 Bootstrap 计算 C-index 的置信区间...")
    for i in range(n_iterations):
        indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)
        sub_y_true = y_true[indices].astype(np.int64)
        if len(np.unique(sub_y_true)) < 2:
            continue

        sub_y_prob = y_surv_prob[indices]
        sub_durations = durations_test[indices]

        surv_df = pd.DataFrame(sub_y_prob.reshape(1, -1), index=[target_time])
        ev = EvalSurv(surv_df, sub_durations, sub_y_true, censor_surv='km')
        stats.append(ev.concordance_td())

    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return full_score, lower, upper


def calculate_classification_metrics(df, n_iterations=1000):
    """计算分类指标及其置信区间"""
    y_true = df['label'].values
    y_scores = df['pos_probs'].values
    y_pred = df['prediction'].values

    # 完整测试集的结果
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
        if len(np.unique(sub_true)) < 2:
            continue

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 配置参数
    model_path = "/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/lung1/pull_strength/pull_strength_pull_0.80/generate/checkpoint-best.pth"
    train_csv_path = "/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/train/final_complete_with_paths_labels.csv"  # 训练集路径（用于计算cutoff）
    test_csv_path = "/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/lung1/test_val/final_complete_with_paths_labels.csv"  # 测试集路径（用于评估）

    # 准备数据
    print("准备数据...")
    data_loader_train, len_train = prepare_dataset(train_csv_path)
    data_loader_test, len_test = prepare_dataset(test_csv_path)
    print(f"训练集样本数: {len_train}, 测试集样本数: {len_test}")

    # 实例化模型
    model = Resnet_1D.ResNet1D().to(device)

    # 加载权重
    print(f"正在加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # ========== 用训练集计算最佳阈值 ==========
    all_train_probs = []
    all_train_labels = []

    print("训练集推理，计算最佳阈值...")
    with torch.no_grad():
        for batch in tqdm(data_loader_train):
            features = batch["features"].to(device)
            labels = batch["labels"]

            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].cpu().numpy()

            if labels is not None:
                label_values = labels if isinstance(labels, np.ndarray) else np.array(labels)
                all_train_probs.extend(pos_probs)
                all_train_labels.extend(label_values)

    # 计算约登指数最佳阈值
    best_threshold = 0.5
    if len(all_train_labels) > 0:
        fpr, tpr, thresholds = roc_curve(all_train_labels, all_train_probs)
        youden_indices = tpr - fpr
        best_idx = np.argmax(youden_indices)
        best_threshold = thresholds[best_idx]
        print(f"最佳阈值（约登指数）: {best_threshold:.4f}")
        print(f"对应灵敏度: {tpr[best_idx]:.4f}, 特异度: {1 - fpr[best_idx]:.4f}")
    else:
        print("未检测到真实标签，使用默认阈值 0.5")

    #
    # best_threshold = 0.5
    # if len(all_train_labels) > 0:
    #     from sklearn.metrics import f1_score
    #     fpr, tpr, thresholds = roc_curve(all_train_labels, all_train_probs)
    #
    #     # 计算每个阈值对应的 F1 分数
    #     f1_scores = []
    #     for thresh in thresholds:
    #         preds = (np.array(all_train_probs) >= thresh).astype(int)
    #         f1 = f1_score(all_train_labels, preds, zero_division=0)
    #         f1_scores.append(f1)
    #
    #     best_idx = np.argmax(f1_scores)
    #     best_threshold = thresholds[best_idx]
    #     print(f"最佳阈值（最大F1）: {best_threshold:.4f}")
    #     print(f"对应F1分数: {f1_scores[best_idx]:.4f}")
    # else:
    #     print("未检测到真实标签，使用默认阈值 0.5")
    #
    # best_threshold = 0.5
    # if len(all_train_labels) > 0:
    #     from sklearn.metrics import matthews_corrcoef
    #     fpr, tpr, thresholds = roc_curve(all_train_labels, all_train_probs)
    #
    #     # 计算每个阈值对应的 MCC
    #     mcc_scores = []
    #     for thresh in thresholds:
    #         preds = (np.array(all_train_probs) >= thresh).astype(int)
    #         mcc = matthews_corrcoef(all_train_labels, preds)
    #         mcc_scores.append(mcc)
    #
    #     best_idx = np.argmax(mcc_scores)
    #     best_threshold = thresholds[best_idx]
    #     print(f"最佳阈值（最大MCC）: {best_threshold:.4f}")
    #     print(f"对应MCC: {mcc_scores[best_idx]:.4f}")
    # else:
    #     print("未检测到真实标签，使用默认阈值 0.5")

    # best_threshold = 0.5
    # if len(all_train_labels) > 0:
    #     fpr, tpr, thresholds = roc_curve(all_train_labels, all_train_probs)
    #
    #     # 计算每个阈值对应的 G-mean
    #     gmean_scores = np.sqrt(tpr * (1 - fpr))
    #
    #     best_idx = np.argmax(gmean_scores)
    #     best_threshold = thresholds[best_idx]
    #     print(f"最佳阈值（最大G-mean）: {best_threshold:.4f}")
    #     print(f"对应G-mean: {gmean_scores[best_idx]:.4f}")
    #     print(f"对应灵敏度: {tpr[best_idx]:.4f}, 特异度: {1 - fpr[best_idx]:.4f}")
    # else:
    #     print("未检测到真实标签，使用默认阈值 0.5")

    # best_threshold = 0.5
    # if len(all_train_labels) > 0:
    #     from sklearn.metrics import fbeta_score
    #     fpr, tpr, thresholds = roc_curve(all_train_labels, all_train_probs)
    #
    #     f05_scores = []
    #     for thresh in thresholds:
    #         preds = (np.array(all_train_probs) >= thresh).astype(int)
    #         f05 = fbeta_score(all_train_labels, preds, beta=0.5, zero_division=0)
    #         f05_scores.append(f05)
    #
    #     best_idx = np.argmax(f05_scores)
    #     best_threshold = thresholds[best_idx]
    #     print(f"最佳阈值（最大F0.5）: {best_threshold:.4f}")
    #     print(f"对应F0.5分数: {f05_scores[best_idx]:.4f}")
    #     print(
    #         f"对应Precision: {precision_score(all_train_labels, (np.array(all_train_probs) >= best_threshold).astype(int), zero_division=0):.4f}")
    #     print(
    #         f"对应Recall: {recall_score(all_train_labels, (np.array(all_train_probs) >= best_threshold).astype(int), zero_division=0):.4f}")
    # else:
    #     print("未检测到真实标签，使用默认阈值 0.5")

    # ========== 测试集推理并计算指标 ==========
    all_test_probs = []
    all_test_labels = []
    all_test_preds = []

    print("测试集推理...")
    with torch.no_grad():
        for batch in tqdm(data_loader_test):
            features = batch["features"].to(device)
            labels = batch["labels"]

            outputs = model(features)
            probs = F.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].cpu().numpy()

            # 使用训练集阈值生成预测
            preds = (pos_probs > best_threshold).astype(int)

            if labels is not None:
                label_values = labels if isinstance(labels, np.ndarray) else np.array(labels)
                all_test_probs.extend(pos_probs)
                all_test_labels.extend(label_values)
                all_test_preds.extend(preds)

    # 构建 DataFrame 用于指标计算
    df_results = pd.DataFrame({
        'label': all_test_labels,
        'pos_probs': all_test_probs,
        'prediction': all_test_preds
    })

    # 计算指标
    print("\n" + "=" * 60)
    print(f"{'指标 (Test Set, threshold from train)':<35} | {'结果 (Score)':<10} | {'95% CI (Bootstrap)':<20}")
    print("-" * 60)

    # C-index
    c_val, c_low, c_high = calculate_c_index_with_ci(df_results)
    print(f"{'C-index (pycox)':<35} | {c_val:.4f}     [{c_low:.4f} - {c_high:.4f}]")

    # 分类指标
    clf_metrics = calculate_classification_metrics(df_results)
    display_map = {
        'auc': 'AUC (ROC)', 'acc': 'Accuracy', 'f1': 'F1-Score',
        'pre': 'Precision', 'recall': 'Recall'
    }

    for key, name in display_map.items():
        score, low, high = clf_metrics[key]
        print(f"{name:<35} | {score:.4f}     [{low:.4f} - {high:.4f}]")

    print("=" * 60)
    print(f'评估总耗时: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')

    # 清理 DataLoader
    del data_loader_train
    del data_loader_test


if __name__ == '__main__':
    start_time = time.time()
    main()