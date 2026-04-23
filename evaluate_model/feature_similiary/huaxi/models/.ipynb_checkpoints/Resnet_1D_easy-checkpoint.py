import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = self.relu(out + residual)
        return out


class ResNet1D(nn.Module):
    def __init__(self, input_dim=400, num_classes=2):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 256)
        self.bn_in = nn.BatchNorm1d(256)

        self.blocks = nn.Sequential(
            ResidualBlock1D(256),
            ResidualBlock1D(256),
            ResidualBlock1D(256)
        )

        self.fc_out = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_in(self.fc_in(x)))
        x = self.blocks(x)
        x = self.fc_out(x)
        return x