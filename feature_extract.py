import argparse
from model.swin3d import Swin3D,Swin3DforPretrain
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from utils.dataset_clip import dataset_clip_cls, dataset_clip_cox
from torch.utils.data import DataLoader
import os
import tqdm
import hashlib
import csv

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--trainpath", default="", type=str, help="directory to train data")
    parser.add_argument("--seed", default=3407, type=int, help="random seed")
    parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
    parser.add_argument('--datadir', type=str, default='none')
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='/home/user/prognosis_lst/feature/')
    parser.add_argument("--freeze", default="all+feature", type=str, help="if freeze")
    parser.add_argument("--pretrained", default="/home/user/prognosis_lst/pretrain/checkpoint-544000/model.safetensors", type=str, help="if pretrained")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument("--device", default="0,1,2,3", type=str, help="gpu device")
    parser.add_argument("--num_classes", default=2, type=int, help="number of classes for multi-classification")
    # 新增：Cox模型时间点AUC评估参数
    parser.add_argument('--img_aug','--img_augmentation', type=bool, default=False)
    parser.add_argument("--head", default="default", type=str, help="cls head")
    parser.add_argument("--testpath", default="/home/user/prognosis_lst/data/dalina/train_for_radiomic2.csv", type=str, help="comma-separated directories to test data")
    parser.add_argument("--tepath_name", default="image", type=str, help="testlabel name in csv")
    parser.add_argument("--telabel_name", default="text", type=str, help="testlabel name in csv")
    parser.add_argument("--oversample", default=False, type=bool, help="if oversample")
    parser.add_argument('--img_size', type=int, nargs='+', default=[48, 256, 256])
    parser.add_argument('--standard_func', type=str, default='max', help='zscore, minmax, none')
    args = parser.parse_args()
    return args




def main(args):


    accelerator = Accelerator()

    model = Swin3D(args=args)
    print('model:', model)


    test_datasets = dataset_clip_cls(args=args, csvpath=args.testpath, mode='test')


    test_loaders = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False,
                               pin_memory=args.pin_memory, num_workers=args.num_workers)

    model = accelerator.prepare(model)
    test_loader = accelerator.prepare(test_loaders)

    model.eval()


    test_path = args.testpath.replace('.csv', '') # 去后缀

    last_two = '/'.join(test_path.split('/')[-2:])  # 'dalina/train_for_radiomic2'

    save_dir = os.path.join(args.save_path,last_two)

    os.makedirs(save_dir, exist_ok=True)


    process_csv = os.path.join(save_dir,f'processed_rank_{accelerator.process_index}.csv')
    processed_paths = set()

    if os.path.exists(process_csv):
        with open(process_csv, 'r') as f:
            for line in f:
                processed_paths.add(line.strip())
    else:
        os.makedirs(save_dir, exist_ok=True)
        with open(process_csv, 'w') as f:
            pass  # 创建空文件
        print(f"创建新文件: {process_csv}")


    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="提取特征"):
            images = batch[0].to(accelerator.device)
            paths = batch[-1]
            features = model(images)
            features = features.detach().cpu()

            # 读取已处理的路径,将之前csv加载的读取进去

            # 追加新路径
            with open(process_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                for feat, path in zip(features, paths):
                    path_str = str(path)
                    md5_hash = hashlib.md5(path_str.encode()).hexdigest()

                    if md5_hash not in processed_paths:
                        # 保存特征文件
                        save_path = os.path.join(save_dir, f'{md5_hash}.pt')
                        torch.save(feat, save_path)

                        # 保存 MD5 和原始路径到 CSV
                        writer.writerow([md5_hash, path_str])
                        f.flush()

                        processed_paths.add(md5_hash)


if __name__ == "__main__":
    args = parse_args()
    main(args)