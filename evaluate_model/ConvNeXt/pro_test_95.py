import argparse
import datetime
import time
import torch
from engine import test
from datasets_test import prepare_dataset
from models import Resnet_1D
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv

def c_index(df,n_iterations=1000):
    stats = []
    y_true = df['label'].values
    y_pred = df['prediction'].values

    for i in range(n_iterations):
        indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)

        score = roc_auc_score(y_true[indices], y_pred[indices])
        stats.append(score)

    alpha = 0.95
    lower = np.percentile(stats, ((1.0 - alpha) / 2.0) * 100)  # 2.5
    upper = np.percentile(stats, (alpha + ((1.0 - alpha) / 2.0)) * 100)  # 97.5

    mean_score = roc_auc_score(y_true, y_pred)

    return mean_score, lower, upper


def calculate_95_ci(df, n_iterations=1000):
    stats = []
    y_true = df['label'].values
    # 注意：此处应使用预测概率，例如 df['prob']，而不是分类后的标签
    y_scores = df['pos_probs'].values

    for i in range(n_iterations):
        # 对索引进行有放回抽样 (Bootstrap)
        indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)

        # 提取抽样后的数据
        sub_y_true = y_true[indices]
        sub_y_scores = y_scores[indices]

        # 检查子集中是否同时包含两类（AUC 要求至少有一个正样本和一个负样本）
        if len(np.unique(sub_y_true)) < 2:
            continue

        score = roc_auc_score(sub_y_true, sub_y_scores)
        stats.append(score)

    # 计算 95% 置信区间
    alpha = 0.95
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)

    # 计算全量数据的平均 AUC
    mean_score = roc_auc_score(y_true, y_scores)

    return mean_score, lower, upper

    stats = []
    y_true = df['label'].values
    y_pred = df['prediction'].values

    for i in range(n_iterations):
        indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)

        score = roc_auc_score(y_true[indices], y_pred[indices])
        stats.append(score)

    alpha = 0.95
    lower = np.percentile(stats, ((1.0 - alpha) / 2.0) * 100)  # 2.5
    upper = np.percentile(stats, (alpha + ((1.0 - alpha) / 2.0)) * 100)  # 97.5

    mean_score = roc_auc_score(y_true, y_pred)

    return mean_score, lower, upper



def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Per GPU batch size')


    # * Finetuning params
    parser.add_argument('--finetune', default='/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/huaxi/normal/base/checkpoint-best.pth',
                        help='finetune from checkpoint')


    parser.add_argument('--test_path', type=str,
                        default="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/val/final_complete_with_paths_labels.csv",
                        help='验证集CSV文件路径')
    # 数据列名参数
    parser.add_argument('--column_names', type=str, default="feature_path",
                        help='特征文件路径的列名')
    parser.add_argument('--labels_name', type=str, default="label",
                        help='特征文件路径的列名')


    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--pin_mem",
        type=bool,
        default=True,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )


    return parser

def main(args):

    device = torch.device("cuda")
    data_loader_test, len_dataset_test = prepare_dataset(args)

    # model = FC.FC(input_dim=400, num_classes=2)
    model = Resnet_1D.ResNet1D()
    # model = MLP.MLPClassifier(input_dim=400, num_classes=2)

    if args.finetune:
        ckpt0 = torch.load(args.finetune, map_location='cpu')
        ckpt0 = ckpt0["model"]
        new_dict = {}
        for k,v in ckpt0.items():
                # print(k,v.shape)
                if 'model.' in k:
                    k = k.replace('model.','')
                    new_dict[k] = v
                else:
                    new_dict[k] = v
        newdict2 = {}
        for k,v in model.state_dict().items():
            # print(k,v.shape)
            if k in new_dict.keys():
                newdict2[k] = new_dict[k]
            else:
                newdict2[k] = v
                print("pretrained.state_dict().keys() not have",k)
        model.load_state_dict(newdict2, strict=True)
        print(f"load lcc pretrained model done from {args.finetune}")

    model.to(device)
    print("Model = %s" % str(model))


    print("Number of test examples = %d" % len_dataset_test)

    print("Start test" )
    start_time = time.time()


    df_results = test(data_loader_test, model, device)
    mean_auc, lower_ci, upper_ci = calculate_95_ci(df_results)

    mean_c_index, lower_c_index, upper_c_index = c_index(df_results)
    
    print(f"准确率 Mean Auc: {mean_auc:.4f}")
    print(f"95% 置信区间 (95% CI): [{lower_ci:.4f}, {upper_ci:.4f}]")
    print(f"准确率 Mean C-index: {mean_c_index:.4f}")
    print(f"95% 置信区间 (95% CI): [{lower_c_index:.4f}, {upper_c_index:.4f}]")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
