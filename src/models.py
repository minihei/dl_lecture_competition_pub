import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int, 
        in_channels: int,
        hid_dim: int = 128,
        dropout_rate: float = 0.5, #追加
        weight_decay: float = 1e-4  #追加
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            nn.Dropout(dropout_rate),  #追加
            ConvBlock(hid_dim, hid_dim),
            nn.Dropout(dropout_rate),  #追加
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )
        self.dropout_rate = dropout_rate #追加
        self.weight_decay = weight_decay #追加

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): *description*
        Returns:
            X ( b, num_classes ): *description*
        """
        X = self.blocks(X)
        return self.head(X)

    def regularization_loss(self): #追加
        # L2正則化の損失を計算 #追加
        l2_reg = 0 #追加
        for param in self.parameters(): #追加
            l2_reg += torch.sum(param ** 2) #追加
        return self.weight_decay * 0.5 * l2_reg #追加



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
    
# class ImprovedConvClassifier(nn.Module):
#     def __init__(self, num_classes, seq_len, num_channels, weight_decay=0.001):
#         super(ImprovedConvClassifier, self).__init__()
#         self.weight_decay = weight_decay

#         self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

#         # 入力データの長さに応じて全結合層の入力サイズを計算
#         test_input = torch.randn(1, num_channels, seq_len)
#         flattened_size = self._get_flattened_size(test_input)

#         self.fc1 = nn.Linear(flattened_size, 512)
#         self.fc2 = nn.Linear(512, num_classes)

#     def _get_flattened_size(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         return x.view(x.size(0), -1).size(1)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

#     def regularization_loss(self):
#         l2_reg = 0
#         for param in self.parameters():
#             l2_reg += torch.sum(param ** 2)
#         return self.weight_decay * 0.5 * l2_reg