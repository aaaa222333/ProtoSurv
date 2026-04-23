import torch
from PIL import Image
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.distributed as dist

def prepare_dataset(args, num_tasks=1, global_rank=0):

    dataset = load_dataset(
        'csv',
        data_files={"train": args.train_path, "val": args.val_path}
    )

    feature_path = args.column_names
    labels_name = args.labels_name



    def preprocess(examples):
        features = [torch.load(feature, weights_only=True) for feature in examples[feature_path]]

        normalized_features = []
        for feature in features:
            # 假设feature范围是 [0, 1]
            normalized = (feature - 0.5) / 0.5  # 映射到 [-1, 1]
            # 或者标准化 (mean=0, std=1)
            # normalized = (feature - feature.mean()) / feature.std()
            normalized_features.append(normalized)

        examples["features"] = normalized_features
        return examples

    train_dataset = dataset["train"].with_transform(preprocess)
    val_dataset = dataset["val"].with_transform(preprocess)

    def collate_fn(examples):
        features = torch.stack([example["features"] for example in examples])
        features = features.to(memory_format=torch.contiguous_format).float()
        labels = torch.tensor([example[labels_name] for example in examples])
        return {"features": features,"labels": labels }



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
