import torch
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset

def prepare_dataset(args):

    dataset = load_dataset(
        'csv',
        data_files={"test": args.test_path}
    )

    feature_path = args.column_names
    labels_name = args.labels_name
    
    def preprocess(examples):
        features = [torch.load(feature, weights_only=True) for feature in examples[feature_path]]

        normalized_features = []
        for feature in features:
            normalized = (feature - 0.5) / 0.5  # 映射到 [-1, 1]
            normalized_features.append(normalized)

        examples["features"] = normalized_features
        return examples

    test_dataset = dataset["test"].with_transform(preprocess)


    def collate_fn(examples):
        features = torch.stack([example["features"] for example in examples])
        features = features.to(memory_format=torch.contiguous_format).float()
        labels = [example[labels_name] for example in examples]
        return {"features": features,"labels": labels}

    sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        sampler=sampler_test,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader_test, len(test_dataset)
