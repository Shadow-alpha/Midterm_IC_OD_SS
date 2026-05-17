import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid_mask = targets != self.ignore_index
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        targets_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss(
    name: str,
    num_classes: int,
    ignore_index: int = 255,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> nn.Module:
    name = name.lower()
    if name in {"ce", "cross_entropy", "cross-entropy"}:
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    if name in {"ce_dice", "combined", "cross_entropy_dice"}:
        return CombinedLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )
    raise ValueError(f"Unsupported loss: {name}")
