import argparse
import datetime
import time
import torch
from engine import test
from datasets_path import prepare_dataset
from models import Resnet_1D_easy
from models import FC


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Per GPU batch size')


    # * Finetuning params
    parser.add_argument('--finetune', default='/home/user/prognosis_lst/evaluate_model/ConvNeXt/output/huaxi/sim/checkpoint-best.pth',
                        help='finetune from checkpoint')


    parser.add_argument('--test_path', type=str,
                        default="/home/user/prognosis_lst/evaluate_model/feature_similiary/huaxi/pseudo_label_results.csv",
                        help='验证集CSV文件路径')

    # parser.add_argument('--test_path', type=str,
    #                     default="/home/user/prognosis_lst/evaluate_model/ConvNeXt/data/huaxi/val/final_complete_with_paths_ori_test.csv",
    #                     help='验证集CSV文件路径')



    # 数据列名参数
    parser.add_argument('--column_names', type=str, default="feature_path",
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

    model = FC.FC(input_dim=400, num_classes=2)
    # model = Resnet_1D_easy.ResNet1D()
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


    df = test(data_loader_test, model, device)
    df.to_csv('inference_results.csv', index=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
