import torch
from datasets import load_dataset
import sys
from torchvision import transforms


## 优先导入库文件
class Args:
    """简单的args类，替代字典"""

    def __init__(self):
        self.train_path = "/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/dalina/train/final_complete_with_paths_labels.csv"  # 改成你的实际路径
        self.val_path = "/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/dalina/val/final_complete_with_paths_labels.csv"  # 改成你的实际路径
        self.column_names = "feature_path"  # 特征文件路径的列名
        self.labels_name = "label"  # 标签列名
        self.batch_size = 32
        self.num_workers = 4
        self.pin_mem = True
        self.seed = 42


def prepare_dataset(args, num_tasks=1, global_rank=0):

    dataset = load_dataset(
        'csv',
        data_files={"train": args.train_path, "val": args.val_path}
    )

    feature_path = args.column_names
    labels_name = args.labels_name



    def preprocess(examples):
        features = [torch.load(feature) for feature in examples[feature_path]]

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
        drop_last=True,
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


def inspect_dataloader(data_loader, name="训练"):
    """测试DataLoader是否能正常迭代"""
    print(f"\n" + "=" * 60)
    print(f"测试{name}DataLoader...")
    print("=" * 60)

    try:
        for i, batch in enumerate(data_loader):
            if batch is None:
                print(f"Batch {i} 为空")
                continue

            features = batch['features']
            labels = batch['labels']

            print(f"Batch {i}:")
            print(f"  特征形状: {features.shape}")
            print(f"  特征类型: {features.dtype}")
            print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")
            print(f"  标签形状: {labels.shape}")
            print(f"  标签: {labels[:5]}")  # 显示前5个标签

            if i >= 2:  # 只测试前3个batch
                break

        print(f"\n✅ {name}DataLoader 测试通过!")

    except Exception as e:
        print(f"❌ {name}DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 创建args对象
    args = Args()


    # 打印配置
    print("配置参数:")
    print(f"  训练集路径: {args.train_path}")
    print(f"  验证集路径: {args.val_path}")
    print(f"  特征列名: {args.column_names}")
    print(f"  标签列名: {args.labels_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")

    # 准备数据集
    train_loader, val_loader, train_size, val_size = prepare_dataset(
        args, num_tasks=1, global_rank=0
    )


    # 测试DataLoader
    inspect_dataloader(train_loader, "训练")
    inspect_dataloader(val_loader, "验证")