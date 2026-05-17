"""
ResNet-18 + CBAM (Convolutional Block Attention Module).
CBAM sequentially applies Channel Attention and Spatial Attention.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class ChannelAttention(nn.Module):
    """Channel Attention module: uses both avg-pooled and max-pooled features."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(x.mean(dim=[2, 3])).view(b, c, 1, 1)
        max_out = self.mlp(x.flatten(start_dim=2).max(dim=2)[0]).view(b, c, 1, 1)
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention module: concatenates avg-pool and max-pool along channel dim."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAMLayer(nn.Module):
    """CBAM: Channel Attention → Spatial Attention."""
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channel, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class CBAMBasicBlock(BasicBlock):
    """BasicBlock with CBAM inserted after the residual addition."""
    def __init__(self, *args, cbam_reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        out_channels = self.conv2.out_channels
        self.cbam = CBAMLayer(out_channels, reduction=cbam_reduction)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.cbam(out)
        return out


def cbam_resnet18(num_classes=102, pretrained=True):
    """ResNet-18 with CBAM blocks inserted."""
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    def _replace_block(layer):
        for name, child in layer.named_children():
            if isinstance(child, BasicBlock):
                cbam_block = CBAMBasicBlock(
                    child.conv1.in_channels,
                    child.conv1.out_channels,
                    stride=child.conv1.stride[0] if hasattr(child.conv1, 'stride') else 1,
                    downsample=child.downsample,
                )
                cbam_block.load_state_dict(child.state_dict(), strict=False)
                setattr(layer, name, cbam_block)

    _replace_block(model.layer1)
    _replace_block(model.layer2)
    _replace_block(model.layer3)
    _replace_block(model.layer4)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
