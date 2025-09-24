import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class PointMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = SharedMLP(3, 64)
        self.mlp2 = SharedMLP(64, 128)
        self.mlp3 = SharedMLP(128, 256)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B x N x 3 → B x 3 x N
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        x = self.pool(x).squeeze(-1)  # B x 256
        x = self.dp1(F.relu(self.bn1(self.fc1(x))))
        x = self.dp2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)
