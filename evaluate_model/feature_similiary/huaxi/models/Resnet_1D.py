import torch.nn as nn
import torch
class ResidualBlock1D(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)  # 扩展维度
        self.bn1 = nn.BatchNorm1d(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)  # 恢复维度
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # 可选：添加SE模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        # SE模块
        se_weight = self.se(out.unsqueeze(-1)).unsqueeze(-1)
        out = out * se_weight.squeeze(-1)
        
        out = self.relu(out + residual)
        out = self.dropout(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, input_dim=400, num_classes=2, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 多尺度特征提取
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock1D(hidden_dims[0], dropout=0.1),
                ResidualBlock1D(hidden_dims[0], dropout=0.1)
            ),
            nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(inplace=True),
                ResidualBlock1D(hidden_dims[1], dropout=0.15),
                ResidualBlock1D(hidden_dims[1], dropout=0.15)
            ),
            nn.Sequential(
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.BatchNorm1d(hidden_dims[2]),
                nn.ReLU(inplace=True),
                ResidualBlock1D(hidden_dims[2], dropout=0.2),
                ResidualBlock1D(hidden_dims[2], dropout=0.2)
            )
        ])
        
        # 全局特征融合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[2] * 2, hidden_dims[3]),  # 拼接全局和局部特征
            nn.BatchNorm1d(hidden_dims[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[3], num_classes)
        )
        
        # 辅助分类器（可选）
        self.aux_classifier = nn.Linear(hidden_dims[2], num_classes)
        
    def forward(self, x, return_features=False):
        # 输入投影
        x = self.input_proj(x)
        
        # 多尺度特征
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        # 全局特征
        global_feat = self.global_pool(x.unsqueeze(-1)).squeeze(-1)
        
        # 局部特征（最后一个block的输出）
        local_feat = x
        
        # 特征融合
        combined = torch.cat([global_feat, local_feat], dim=1)
        
        # 分类
        out = self.classifier(combined)
        
        if return_features:
            return out, combined
        return out