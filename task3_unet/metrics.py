import torch


class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1).detach().cpu()
        targets = targets.detach().cpu()
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)

        preds = preds[valid]
        targets = targets[valid]
        if targets.numel() == 0:
            return

        encoded = targets * self.num_classes + preds
        hist = torch.bincount(encoded, minlength=self.num_classes ** 2)
        hist = hist.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

    def pixel_accuracy(self) -> float:
        total = self.confusion_matrix.sum().item()
        if total == 0:
            return 0.0
        correct = torch.diag(self.confusion_matrix).sum().item()
        return correct / total

    def miou(self) -> float:
        hist = self.confusion_matrix.float()
        intersection = torch.diag(hist)
        union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
        valid = union > 0
        if not torch.any(valid):
            return 0.0
        return (intersection[valid] / union[valid]).mean().item()

    def compute(self) -> dict[str, float]:
        return {
            "pixel_acc": self.pixel_accuracy(),
            "miou": self.miou(),
        }
