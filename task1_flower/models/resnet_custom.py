"""
ResNet-18/34 with modified output layer for 102-class flower classification.
"""
import torch.nn as nn
from torchvision import models


def resnet18(num_classes=102, pretrained=True):
    """ResNet-18 with output layer replaced for 102 classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def resnet34(num_classes=102, pretrained=True):
    """ResNet-34 with output layer replaced for 102 classes."""
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
