import torch
from PIL import Image
from datasets import Features, Value, Dataset, DatasetDict, IterableDatasetDict, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.distributed as dist
import numpy as np
import os

# feature0 = torch.load("/home/user/prognosis_lst/evaluate_model/feature_similiary/data/lung1/class_0_average.pt")
# feature1 = torch.load("/home/user/prognosis_lst/evaluate_model/feature_similiary/data/lung1/class_1_average.pt")
feature0 = torch.load("/home/user/prognosis_lst/evaluate_model/feature_similiary/data/huaxi/class_0_average.pt")
feature1 = torch.load("/home/user/prognosis_lst/evaluate_model/feature_similiary/data/huaxi/class_1_average.pt")

# stats0 = {'mean': 0.680536, 'std': 0.200304, 'q1': 0.608118, 'q3': 0.815525}
# stats1 = {'mean': 0.768731, 'std': 0.120858, 'q1': 0.706562, 'q3': 0.854699}


stats0 = {
    'mean': 0.832611, 'std': 0.143141, 'q1': 0.785900, 'q3': 0.917102
}
stats1 = {
    'mean': 0.787413, 'std': 0.295031, 'q1': 0.792394, 'q3': 0.926512
}  # 假设类别1有类似统计


def calculate_thresholds(stats0, stats1, method='percentile'):
    if method == 'percentile':
        threshold1_0 = stats0['q3']  # 类别0的高阈值
        threshold2_0 = stats0['q1']  # 类别0的低阈值
        threshold1_1 = stats1['q3']  # 类别1的高阈值
        threshold2_1 = stats1['q1']  # 类别1的低阈值
    elif method == 'mean_std':
        threshold1_0 = stats0['mean'] + 0.5 * stats0['std']
        threshold2_0 = stats0['mean'] - 0.5 * stats0['std']
        threshold1_1 = stats1['mean'] + 0.5 * stats1['std']
        threshold2_1 = stats1['mean'] - 0.5 * stats1['std']
    return threshold1_0, threshold1_1, threshold2_0, threshold2_1


th1_0, th1_1, th2_0, th2_1 = calculate_thresholds(stats0, stats1, method='percentile')

threshold = {
    0: (th1_0, th2_0),
    1: (th1_1, th2_1)  # 注意这里是 th1_1，不是 th1_0
}

feature_center = {
    0: feature0,
    1: feature1
}

my_features = Features({
    'feature_path': Value("string"),
    'label': Value("int64"),  # 标签必须是整数
    'need_pull': Value("bool"),
    'sim_to_pseudo': Value("float64")  # 强制这一列为浮点数
})


def feature_pull(feature, target_feature, similarity, threshold1, threshold2, pull_strength=0.5):
    """
    Args:
        feature: 当前特征
        target_feature: 目标类别中心特征
        similarity: 当前相似度
        threshold: 阈值2
        pull_strength: 拉近强度（0-1）
    """
    # 计算需要拉近的程度（相似度离阈值越远，拉得越多）pull这个值最低为0，最高可能为0.815-0.60 = 0.215/0.4=0.5，，当pull=1时，限制在了0和0.5直接，
    pull_factor = (threshold1 - similarity) / (threshold1 - threshold2) * pull_strength
    pull_factor = np.clip(pull_factor, 0, 0.5)  # 限制最大拉近程度

    # 加权平均拉近
    pulled_feature = (1 - pull_factor) * feature + pull_factor * target_feature

    # pulled_feature = feature
    #
    return pulled_feature


## lung1 0.4，huaxi0.5


def prepare_dataset(args, num_tasks=1, global_rank=0):
    data_files = {"train": args.train_path, "val": args.val_path}

    if args.pre_path != "":
        if os.path.exists(args.pre_path):
            data_files["pre_path"] = args.pre_path

    dataset = load_dataset(
        'csv',
        data_files=data_files,
        features=my_features
    )

    feature_path = args.column_names
    labels_name = args.labels_name

    # def preprocess(examples):
    #     normalized_features = []
    #     for example in examples:
    #         feature = torch.load(example[feature_path], weights_only=True)
    #         if example["need_pull"] == True:
    #             feature = feature_pull(feature, feature_center[example["label"]],example["sim_to_pseudo"],threshold[example["label"]][0],threshold[example["label"]][1])
    #         else:
    #             feature = feature
    #         normalized = (feature - 0.5) / 0.5  # 映射到 [-1, 1]
    #         normalized_features.append(normalized)
    #     examples["features"] = normalized_features
    #     return examples

    def preprocess(examples):
        normalized_features = []
        n_samples = len(examples[feature_path])

        for i in range(n_samples):
            # 获取当前样本信息
            path = examples[feature_path][i]

            need_pull = examples['need_pull'][i]
            label = examples['label'][i]
            sim = examples['sim_to_pseudo'][i]
            # 加载特征
            feature = torch.load(path, weights_only=True)

            # 拉近处理
            if need_pull and label in feature_center:
                th1, th2 = threshold[label]
                feature = feature_pull(feature, feature_center[label], sim, th1, th2, args.pull_strength)

            # 标准化
            normalized = (feature - 0.5) / 0.5
            normalized_features.append(normalized)

        examples["features"] = normalized_features
        return examples

    train_dataset = dataset["train"].with_transform(preprocess)
    if "pre_path" in dataset:
        pre_dataset = dataset["pre_path"].with_transform(preprocess)
        train_dataset = concatenate_datasets([train_dataset, pre_dataset])

    val_dataset = dataset["val"].with_transform(preprocess)

    def collate_fn(examples):
        features = torch.stack([example["features"] for example in examples])
        features = features.to(memory_format=torch.contiguous_format).float()
        labels = torch.tensor([example[labels_name] for example in examples])
        return {"features": features, "labels": labels}

    # 创建采样器
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
        seed=args.seed,
    )

    sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    # 创建数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler_val,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader_train, data_loader_val, len(train_dataset), len(val_dataset)



