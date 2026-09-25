import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SPPF_SE(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer with SE attention for YOLOv5."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.se = SEBlock(c_ * 4)

    def forward(self, x):
        """Forward pass through SPPF with SE attention."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        # Apply SE attention to the concatenated features
        out = torch.cat((x, y1, y2, self.m(y2)), 1)
        out = self.se(out)
        return self.cv2(out)
