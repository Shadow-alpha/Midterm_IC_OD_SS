import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from datasets import build_dataloaders
from losses import build_loss
from metrics import SegmentationMetric
from models import UNet
from utils import CSVLogger, get_device, load_config, set_seed


def print_config(cfg: dict, indent: int = 0) -> None:
    prefix = " " * indent
    for key, value in cfg.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 2)
        else:
            print(f"{prefix}{key}: {value}")


def format_weight(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_run_name(cfg: dict) -> str:
    loss_name = cfg["train"]["loss"]
    if loss_name == "ce_dice":
        ce_weight = format_weight(cfg["train"]["ce_weight"])
        dice_weight = format_weight(cfg["train"]["dice_weight"])
        return f"{loss_name}_ce{ce_weight}_dice{dice_weight}"
    return loss_name


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    num_classes: int,
    ignore_index: int,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        metric.update(logits, targets)

    return total_loss / len(loader.dataset), metric.compute()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        metric.update(logits, targets)

    return total_loss / len(loader.dataset), metric.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net on Stanford Background Dataset.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--loss", default=None, choices=["ce", "dice", "ce_dice"], help="Override loss.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--ce-weight", type=float, default=None, help="CE weight for ce_dice loss.")
    parser.add_argument("--dice-weight", type=float, default=None, help="Dice weight for ce_dice loss.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.loss is not None:
        cfg["train"]["loss"] = args.loss
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["learning_rate"] = args.lr
    if args.ce_weight is not None:
        cfg["train"]["ce_weight"] = args.ce_weight
    if args.dice_weight is not None:
        cfg["train"]["dice_weight"] = args.dice_weight

    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    use_amp = bool(cfg["train"]["amp"] and device.type == "cuda")
    print("Training configuration:")
    print_config(cfg, indent=2)
    print(f"  resolved_device: {device}")
    print(f"  resolved_amp: {use_amp}")

    train_loader, val_loader = build_dataloaders(
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

    criterion = build_loss(
        cfg["train"]["loss"],
        num_classes=cfg["data"]["num_classes"],
        ignore_index=cfg["data"]["ignore_index"],
        ce_weight=cfg["train"]["ce_weight"],
        dice_weight=cfg["train"]["dice_weight"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = GradScaler(enabled=use_amp)

    run_name = build_run_name(cfg)
    save_dir = Path(cfg["train"]["save_dir"]) / run_name
    log_dir = Path(cfg["train"]["log_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(
        log_dir / "metrics.csv",
        [
            "epoch",
            "lr",
            "loss/train",
            "loss/val",
            "pixel_acc/train",
            "pixel_acc/val",
            "miou/train",
            "miou/val",
        ],
    )

    wandb_run = None
    if cfg["logging"].get("use_wandb", False):
        import wandb

        wandb_run = wandb.init(
            project=cfg["logging"]["project"],
            name=f"unet-{run_name}",
            config=cfg,
        )

    best_miou = -1.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            num_classes=cfg["data"]["num_classes"],
            ignore_index=cfg["data"]["ignore_index"],
        )
        val_loss, metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=cfg["data"]["num_classes"],
            ignore_index=cfg["data"]["ignore_index"],
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss/train": train_loss,
            "loss/val": val_loss,
            "pixel_acc/train": train_metrics["pixel_acc"],
            "pixel_acc/val": metrics["pixel_acc"],
            "miou/train": train_metrics["miou"],
            "miou/val": metrics["miou"],
        }
        logger.log(row)
        if wandb_run is not None:
            wandb_run.log(row)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"train_acc={train_metrics['pixel_acc']:.4f} | "
            f"train_mIoU={train_metrics['miou']:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={metrics['pixel_acc']:.4f} | "
            f"val_mIoU={metrics['miou']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "train_metrics": train_metrics,
            "val_metrics": metrics,
        }
        torch.save(checkpoint, save_dir / "last.pt")
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save(checkpoint, save_dir / "best.pt")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
