import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out

class SPPF_SelfAttention(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer with Self-Attention for YOLOv5."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.self_attention = SelfAttention(c_ * 4)

    def forward(self, x):
        """Forward pass through SPPF with Self-Attention."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        # Apply Self-Attention to the concatenated features
        out = torch.cat((x, y1, y2, self.m(y2)), 1)
        out = self.self_attention(out)
        return self.cv2(out)
