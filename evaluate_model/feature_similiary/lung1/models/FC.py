import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_dim=400, num_classes=2):
        super().__init__()
        self.net = nn.Linear(input_dim,num_classes)

    def forward(self, x):
        return self.net(x)