import torch
import torch.nn as nn
import torch.nn.functional as F

def get_graph_feature(x, k=20, idx=None):
    # x: B x C x N
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        # Compute pairwise distance and find k-NN
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # B x N x k

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # B x N x C
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # B x 2C x N x k

class DGCNNClassifier(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, num_classes=10):
        super(DGCNNClassifier, self).__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, 1, bias=False), nn.BatchNorm1d(emb_dims), nn.LeakyReLU(0.2))

        self.fc1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        B, N, _ = x.size()
        x = x.transpose(2, 1).contiguous()  # B x 3 x N
        x0 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1)[0]
        x1 = self.conv2(get_graph_feature(x0, k=self.k)).max(dim=-1)[0]
        x2 = self.conv3(get_graph_feature(x1, k=self.k)).max(dim=-1)[0]
        x3 = self.conv4(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]

        x_cat = torch.cat((x0, x1, x2, x3), dim=1)  # B x 512 x N
        x_feat = self.conv5(x_cat)

        x_max = F.adaptive_max_pool1d(x_feat, 1).view(B, -1)
        x_avg = F.adaptive_avg_pool1d(x_feat, 1).view(B, -1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = F.leaky_relu(self.bn6(self.fc1(x)), 0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.fc2(x)), 0.2)
        x = self.dp2(x)
        return self.fc3(x)
