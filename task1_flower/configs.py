"""
Experiment configuration for Task 1: Flower Recognition.

Defines all 7 experiment configurations:
  1. Baseline ResNet-18
  2. Baseline ResNet-34
  3. ResNet-18 varying epochs (15, 50)
  4. ResNet-18 varying LR (1e-2, 1e-4 for head)
  5. ResNet-18 random init (pretrain ablation)
  6. ResNet-18 + SE Block
  7. ResNet-18 + CBAM
"""
from torch import optim
import torch.nn as nn


# Base training params
DATA_ROOT = "./data/flowers102"
BATCH_SIZE = 32
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 102

# Optimizer settings
BASE_LR = 1e-3          # for the new fc layer (head)
BACKBONE_LR = 1e-5      # for pretrained backbone
HEAD_LR_MULTIPLIER = 10  # actually we want head to have higher lr


def get_optimizer_params(model, head_lr=1e-3, backbone_lr=1e-5, weight_decay=1e-4):
    """Split parameters into backbone (pretrained) and head (new fc layer)."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def get_config(config_name):
    """Return the experiment configuration dict."""
    configs = {
        # === Experiment 1: Baseline ResNet-18 ===
        "baseline_r18": {
            "name": "baseline_r18",
            "model": "resnet18",
            "pretrained": True,
            "epochs": 30,
            "head_lr": 1e-3,
            "backbone_lr": 1e-5,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "Baseline: ResNet-18 pretrained, fine-tune 30 epochs",
        },

        # === Experiment 2: Baseline ResNet-34 ===
        "baseline_r34": {
            "name": "baseline_r34",
            "model": "resnet34",
            "pretrained": True,
            "epochs": 30,
            "head_lr": 1e-3,
            "backbone_lr": 1e-5,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "Baseline: ResNet-34 pretrained, fine-tune 30 epochs",
        },

        # === Experiment 3a: Fewer epochs ===
        "epochs_15": {
            "name": "epochs_15",
            "model": "resnet18",
            "pretrained": True,
            "epochs": 15,
            "head_lr": 1e-3,
            "backbone_lr": 1e-5,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "Epoch ablation: 15 epochs",
        },

        # === Experiment 3b: More epochs ===
        "epochs_50": {
            "name": "epochs_50",
            "model": "resnet18",
            "pretrained": True,
            "epochs": 50,
            "head_lr": 1e-3,
            "backbone_lr": 1e-5,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "Epoch ablation: 50 epochs",
        },

        # === Experiment 4a: Higher LR ===
        "lr_high": {
            "name": "lr_high",
            "model": "resnet18",
            "pretrained": True,
            "epochs": 30,
            "head_lr": 1e-2,
            "backbone_lr": 1e-4,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "LR ablation: high LR (head=1e-2, backbone=1e-4)",
        },

        # === Experiment 4b: Lower LR ===
        "lr_low": {
            "name": "lr_low",
            "model": "resnet18",
            "pretrained": True,
            "epochs": 30,
            "head_lr": 1e-4,
            "backbone_lr": 1e-6,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "LR ablation: low LR (head=1e-4, backbone=1e-6)",
        },

        # === Experiment 5: Random init (pretrain ablation) ===
        "random_init": {
            "name": "random_init",
            "model": "resnet18",
            "pretrained": False,
            "epochs": 30,
            "head_lr": 1e-3,
            "backbone_lr": 1e-3,   # same LR for all layers since no pretrain
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "Ablation: ResNet-18 random init, train from scratch",
        },

        # === Experiment 6: ResNet-18 + SE Block ===
        "se_resnet18": {
            "name": "se_resnet18",
            "model": "se_resnet18",
            "pretrained": True,
            "epochs": 30,
            "head_lr": 1e-3,
            "backbone_lr": 1e-5,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "Attention: ResNet-18 + SE Block",
        },

        # === Experiment 7: ResNet-18 + CBAM ===
        "cbam_resnet18": {
            "name": "cbam_resnet18",
            "model": "cbam_resnet18",
            "pretrained": True,
            "epochs": 30,
            "head_lr": 1e-3,
            "backbone_lr": 1e-5,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "description": "Attention: ResNet-18 + CBAM",
        },
    }

    if config_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    return configs[config_name]
