import argparse

import torch
from tqdm import tqdm

from datasets import build_dataloaders
from metrics import SegmentationMetric
from models import UNet
from utils import get_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint, e.g. checkpoints/ce/best.pt")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root

    device = get_device(args.device)
    _, val_loader = build_dataloaders(
        root=cfg["data"]["root"],
        image_size=tuple(cfg["data"]["image_size"]),
        batch_size=cfg["train"]["batch_size"],
        val_ratio=cfg["data"]["val_ratio"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
        ignore_index=cfg["data"]["ignore_index"],
    )

    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["data"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metric = SegmentationMetric(
        num_classes=cfg["data"]["num_classes"],
        ignore_index=cfg["data"]["ignore_index"],
    )
    for images, targets in tqdm(val_loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        metric.update(logits, targets)

    metrics = metric.compute()
    print(f"pixel_acc: {metrics['pixel_acc']:.4f}")
    print(f"mIoU: {metrics['miou']:.4f}")


if __name__ == "__main__":
    main()
