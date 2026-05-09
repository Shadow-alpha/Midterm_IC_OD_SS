"""
ResNet-18 + Squeeze-and-Excitation (SE) Block.
SE block is inserted after each residual block.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class SELayer(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(BasicBlock):
    """BasicBlock with SE layer inserted after the residual addition."""
    def __init__(self, *args, se_reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        out_channels = self.conv2.out_channels
        self.se = SELayer(out_channels, reduction=se_reduction)

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
        out = self.se(out)
        return out


def se_resnet18(num_classes=102, pretrained=True):
    """ResNet-18 with SE blocks inserted."""
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    # Replace BasicBlock layers with SEBasicBlock
    def _replace_block(layer):
        for name, child in layer.named_children():
            if isinstance(child, BasicBlock):
                se_block = SEBasicBlock(
                    child.conv1.in_channels,
                    child.conv1.out_channels,
                    stride=child.conv1.stride[0] if hasattr(child.conv1, 'stride') else 1,
                    downsample=child.downsample,
                )
                # Copy pretrained weights
                se_block.load_state_dict(child.state_dict(), strict=False)
                setattr(layer, name, se_block)

    _replace_block(model.layer1)
    _replace_block(model.layer2)
    _replace_block(model.layer3)
    _replace_block(model.layer4)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
