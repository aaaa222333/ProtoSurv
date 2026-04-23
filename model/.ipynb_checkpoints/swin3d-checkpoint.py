from re import S
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.video import Swin3D_B_Weights
from safetensors.torch import load_file
# warning
import warnings
import argparse
print(torchvision.__file__)





def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--trainpath", default="", type=str, help="directory to train data")
    parser.add_argument("--seed", default=3407, type=int, help="random seed")
    parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
    parser.add_argument('--datadir', type=str, default='none')
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='/home/user/prognosis_lst/feature/')
    parser.add_argument("--freeze", default="all+feature", type=str, help="if freeze")
    parser.add_argument("--pretrained", default='/data1/lcc/log/CLIP/downstream_task/20241219mae2/PFS/v1/6new-swin3d-20clsOri-544000/model_bestVal_val0.pt', type=str, help="if pretrained")
    parser.add_argument("--batch_size", default=40, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--device", default="0,1,2,3", type=str, help="gpu device")
    parser.add_argument("--num_classes", default=2, type=int, help="number of classes for multi-classification")
    # 新增：Cox模型时间点AUC评估参数
    parser.add_argument('--img_aug','--img_augmentation', type=bool, default=False)
    parser.add_argument("--head", default="default", type=str, help="cls head")
    parser.add_argument("--testpath", default="/home/user/prognosis_lst/data/pretrain/train.csv", type=str, help="comma-separated directories to test data")
    parser.add_argument("--tepath_name", default="image", type=str, help="testlabel name in csv")
    parser.add_argument("--telabel_name", default="text", type=str, help="testlabel name in csv")
    parser.add_argument("--oversample", default=False, type=bool, help="if oversample")
    parser.add_argument('--img_size', type=int, nargs='+', default=[48, 256, 256])
    parser.add_argument('--standard_func', type=str, default='max', help='zscore, minmax, none')
    args = parser.parse_args()
    return args






class DifferentiableXGB(nn.Module):
    def __init__(self, input_dim=1024, num_trees=100, tree_depth=3):
        super().__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.trees = nn.ModuleList()
        
        # 初始化100棵可微分决策树（模拟XGBoost的多树集成）
        for _ in range(num_trees):
            tree = nn.Sequential(
                # 树分裂节点：用线性层模拟特征选择和阈值判断
                nn.Linear(input_dim, 2**(tree_depth-1)),  # 每个内部节点对应一个特征+阈值
                nn.Sigmoid(),  # 分裂概率（0/1对应左右子树）
                # 叶节点：输出每个叶节点的权重（模拟XGBoost的叶节点得分）
                nn.Linear(2**(tree_depth-1), 2**tree_depth),
                nn.ReLU()
            )
            self.trees.append(tree)
        
        # 最终集成：加权求和所有树的输出（模拟XGBoost的投票）
        self.final_weight = nn.Parameter(torch.ones(num_trees)/num_trees)  # 树的权重（可训练）
        self.fc = nn.Linear(2*(tree_depth-1), 2)  # 输出类别logits

    def forward(self, x):
        tree_outputs = []
        for tree in self.trees:
            split_probs = tree[0](x)  # (batch_size, 2^(d-1))：每个内部节点的分裂概率
            leaf_weights = tree[1](split_probs)  # (batch_size, 2^d)：叶节点权重
            tree_output = torch.sum(split_probs.unsqueeze(1) * leaf_weights.unsqueeze(2), dim=-1)
            tree_outputs.append(tree_output)
        
        # 集成所有树的输出（加权求和）
        jicheng_output = torch.stack(tree_outputs, dim=1)  # (batch_size, num_trees, 2^d)
        weighted_output = torch.sum(jicheng_output * self.final_weight.unsqueeze(0).unsqueeze(-1), dim=1)
        print("weighted_output:",weighted_output.shape)
        logits = self.fc(weighted_output)
        print("logits:",logits.shape)
        return logits

class Conv1dClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super().__init__()
        # 1D卷积层：输入特征维度1024，输出32个特征图，卷积核大小为3（捕捉相邻3个维度的关联）
        self.conv1 = nn.Conv1d(
            in_channels=1,  # 输入通道数：1（因为特征是(1,1024)，可视为1个通道的序列）
            out_channels=32,  # 卷积核数量（提取32种局部模式）
            kernel_size=3,  # 卷积核长度：每次看3个相邻维度
            stride=1,  # 步长1：不跳过维度
            padding=1  # 补边1：保持卷积后长度不变（1024→1024）
        )
        self.relu = nn.ReLU()
        # 池化层：压缩长度（1024→512），保留重要特征
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 第二个卷积层：进一步提取高级特征
        self.conv2 = nn.Conv1d(
            in_channels=32,  # 输入通道数=上一层输出的32
            out_channels=64,  # 更多特征模式
            kernel_size=3,
            padding=1
        )
        # 全局平均池化：将(64, 256) → (64, 1)，压缩为固定长度向量
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # 最终分类层：将64维特征映射到2个类别
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 输入x形状：(batch_size, 1, 1024)，例如(1,1,1024)
        x = self.conv1(x)  # 输出：(1, 32, 1024)
        x = self.relu(x)
        x = self.pool(x)  # 输出：(1, 32, 512)
        
        x = self.conv2(x)  # 输出：(1, 64, 512)
        x = self.relu(x)
        x = self.pool(x)  # 输出：(1, 64, 256)
        
        x = self.global_pool(x)  # 输出：(1, 64, 1)
        x = x.flatten(1)  # 展平为(1, 64)
        
        x = self.fc(x)  # 输出：(1, 2)（类别分数）
        return x


class Swin3D(nn.Module):
    def __init__(self, num_classes=2, args=None):
        super(Swin3D, self).__init__()
        self.args = args
        print("args.pretrained:",args.pretrained,"debug")
        try:
            self.multi_linear = args.multi_linear
            print("args.multi_linear:", self.multi_linear)
        except:
            self.multi_linear = False
            print("args.multi_linear not found, use default: False")
        
        self.num_classes = args.num_classes

        if self.args.pretrained == 'k400_imagenet' or self.args.pretrained == 'default':
            weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
            print("use kinetics400_imagenet pretrained model")

        elif self.args.pretrained == 'k400':
            weights=Swin3D_B_Weights.KINETICS400_V1
            print("use kinetics400 pretrained model")
        elif self.args.pretrained == 'nopretrained' or self.args.pretrained == 'nopretrain' or self.args.pretrained == 'None':
            weights=None
            print("no pretrained model")
        elif self.args.pretrained == 'none':
            # exit("请确认是否需要预训练权重，为了保险起见不要使用none")
            # raise Warning("请确认是否需要预训练权重，为了保险起见不要使用none")
            weights=None
            print("no pretrained model")
        else:
            weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
            print("pretrained model not found, use default pretrained model: kinetics400_imagenet")
            print("使用默认权重：kinetics400_imagenet")

        self.model =  torchvision.models.video.swin3d_b(num_classes=400, weights=weights)
        
        if self.multi_linear:
            self.head = nn.ModuleList([
                nn.Linear(400, 1, bias=True) for _ in range(num_classes)
            ])
           
        else:
            self.head = nn.Linear(in_features=400, out_features=num_classes, bias=True)
        
        if "all" == self.args.head:
            self.head = nn.Linear(400, num_classes)
            print("freeze all，用400分类头-num_classes分类")
        elif "default" == self.args.head:
            self.head = nn.Linear(400, num_classes)
            print("freeze all，用400分类头-num_classes分类")
        elif 'all+400' == self.args.head:
            self.head = nn.Linear(400, num_classes)
            print("freeze all，用400分类头-num_classes分类")
        elif 'all不用400' == self.args.head:
            self.model.head = nn.Identity()
            self.head = nn.Linear(1024, num_classes)
            print("freeze all，没有用400分类头，直接1024-num_classes分类")
        elif 'all-400' == self.args.head:
            self.model.head = nn.Identity()
            self.head = nn.Linear(1024, num_classes)
            print("freeze all，没有用400分类头，直接1024-num_classes分类")
        else:
            self.head = nn.Linear(400, num_classes)
            print("freeze all，用400分类头-num_classes分类")
        
        
        if 'trans' in self.args.head.lower():
            self.model.head = nn.Identity()
            self.head = nn.ModuleList([nn.TransformerEncoderLayer(d_model=1024, nhead=8), nn.Linear(1024, num_classes)])
        elif 'conv' in self.args.head.lower():
            self.model.head = nn.Identity()
            self.head = Conv1dClassifier(input_dim=1024, num_classes=num_classes)
        elif 'xgb' in self.args.head.lower():
            self.model.head = nn.Identity()
            self.head = DifferentiableXGB()
        

        
        
        if 'lcc' in self.args.pretrained:
            print("load lcc pretrained model")
            new_dict = {}

            # 使用 safetensors 加载模型
            if args.pretrained.endswith('.safetensors'):
                ckpt0 = load_file(self.args.pretrained)
            else:
                ckpt0 = torch.load(self.args.pretrained,weights_only=False)
                ckpt0 = ckpt0["state_dict"]
            print("模型文结构如下")
            for k,v in ckpt0.items():
                print(k,v.shape)
                if 'model.' in k:
                    k = k.replace('model.','')
                    new_dict[k] = v
                else:
                    pass


            newdict2 = {}
            for k,v in self.model.state_dict().items():
                # print(k,v.shape)
                if k in new_dict.keys():
                    newdict2[k] = new_dict[k]
                else:
                    newdict2[k] = v
                    print("pretrained.state_dict().keys() not have",k)
            self.model.load_state_dict(newdict2, strict=True)
            print(f"load lcc pretrained model done from {self.args.pretrained}")


        if args.freeze == 'feature':
            print('只打开patch_embed的梯度和最后的head的梯度')
            self.model.requires_grad_(False)
            self.model.patch_embed.requires_grad_(True)
            self.head.requires_grad_(True)
        elif args.freeze == 'all':
            print('只打开最后的head的梯度')
            self.model.requires_grad_(False)
            self.head.requires_grad_(True)
        elif args.freeze == 'all-400':
            print('打开所有梯度')
            self.model.requires_grad_(False)
            self.model.head.requires_grad_(True)
            self.head.requires_grad_(True)
        elif args.freeze == 'patchembedding':
            print("冻结patchembedding，其他的都打开")
            self.model.patch_embed.requires_grad_(False)
        elif args.freeze == 'all+feature':
            print("梯度全关")
            self.model.requires_grad_(False)


    def forward(self, x):

        x = self.model(x)
        return x


class Swin3DforPretrain(nn.Module):
    def __init__(self, args=None, pretrain_mode='clip'):
        super(Swin3DforPretrain, self).__init__()
        self.pretrain_mode = pretrain_mode
        # self.args = args
        # if self.args.pretrained == 'k400_imagenet':
        #     weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        # elif self.args.pretrained == 'k400':
        #     weights=Swin3D_B_Weights.KINETICS400_V1
        # elif self.args.pretrained == 'nopretrained' or 'nopretrain':
        #     weights=None
        # elif self.args.pretrained == 'none':
        #     exit("请确认是否需要预训练权重，为了保险起见不要使用none")
        # else:
        weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        print("pretrained model not found, use default pretrained model: kinetics400_imagenet")
        print("使用默认权重：kinetics400_imagenet")
        if self.pretrain_mode == 'clip':
            self.model =  torchvision.models.video.swin3d_b(num_classes=400, weights=weights)
            self.model.head = nn.Identity()
            
        elif self.pretrain_mode == 'mae':
            self.model = torchvision.models.video.swin3d_b(num_classes=400,weights=weights)
            self.model.norm = nn.Identity()
            self.model.avgpool = nn.Identity()
            self.model.head = nn.Identity()
            ### 使用卷积将1024*24*8*8 》》 512*6*8*8
            self.convlast =  nn.Conv3d(
                                        in_channels=1024,   # 输入通道数
                                        out_channels=512,   # 输出通道数
                                        kernel_size=(3, 3, 3),  # 三维卷积核
                                        stride=(4, 1, 1),   # 步幅
                                        padding=(1, 1, 1)   # 填充
                                        )

        if args:
            self.args = args
            if args.pretrained:
                print("load lcc pretrained model")
                new_dict = {}

                # 使用 safetensors 加载模型
                ckpt0 = load_file(self.args.pretrained)

                for k,v in ckpt0.items():
                    if 'vision_encoder.model.' in k:
                        k = k.replace('vision_encoder.model.','')
                        new_dict[k] = v
                    else:
                        pass
                
                newdict2 = {}
                for k,v in self.model.state_dict().items():
                    if k in new_dict.keys():
                        newdict2[k] = new_dict[k]
                    else:
                        newdict2[k] = v
                        print(k,"not in self.model.state_dict().keys()")
                self.model.load_state_dict(newdict2, strict=True)


    def forward(self, x):
        if self.pretrain_mode == 'clip':
            x = self.model(x)
        elif self.pretrain_mode == 'mae':
            x = self.model(x)
            x = x.reshape(x.shape[0], 1024, 24, 8, 8)
            ## 使用卷积将1024*24*8*8 》》 512*6*8*8
            x = self.convlast(x)
            

        # print(x.shape)
        # x = self.head(x)
        
        return x


if __name__ == "__main__":
    args = parse_args()
    model = Swin3D(args=args)
    print('model:', model)
 
    
