import torch
import torch.nn as nn
import torch.nn.functional as F

class GAMAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAMAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_global = self.global_pool(x)
        x_global = self.conv(x_global)
        if x_global.size(0) > 1:  # Only apply batch norm if batch size is greater than 1
            x_global = self.bn(x_global)
        x_global = self.relu(x_global)
        out = x * x_global.expand_as(x)
        return out

class SPPF_GAM(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer with GAM attention for YOLOv5."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.gam = GAMAttention(c_ * 4, c_ * 4)

    def forward(self, x):
        """Forward pass through SPPF with GAM attention."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # Apply GAM attention to the concatenated features
        out = torch.cat((x, y1, y2, y3), 1)

        print(f"Shape before GAM: {out.shape}")

        out = self.gam(out)

        print(f"Shape after GAM: {out.shape}")

        return self.cv2(out)

# 使用示例
# model = SPPF_GAM(c1, c2, k)
# for batch in train_dataloader:
#     loss = model.training_step(batch, batch_idx)
#     print(f"Loss: {loss.item()}")
