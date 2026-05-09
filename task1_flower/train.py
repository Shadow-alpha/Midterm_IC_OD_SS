"""
Unified training script for Task 1: Flower Recognition.

Usage:
    # Run a single experiment
    python train.py --config baseline_r18

    # Run all experiments sequentially
    python train.py --all

    # Resume from checkpoint
    python train.py --config baseline_r18 --resume checkpoints/baseline_r18_best.pth
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import wandb

# Ensure print is flushed immediately for log monitoring
print = lambda *args, **kwargs: __import__('builtins').print(*args, **kwargs, flush=True)

# Allow wandb anonymous mode if not logged in
os.environ.setdefault("WANDB_SILENT", "true")
try:
    wandb.login(anonymous="allow", relogin=True)
except Exception:
    os.environ["WANDB_MODE"] = "offline"

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_config, get_optimizer_params, DATA_ROOT, NUM_CLASSES
from models import resnet18, resnet34, se_resnet18, cbam_resnet18
from utils.dataset import get_dataloaders
from utils.train_utils import train_one_epoch, validate, save_checkpoint


MODEL_REGISTRY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "se_resnet18": se_resnet18,
    "cbam_resnet18": cbam_resnet18,
}


def build_model(cfg, device):
    """Build model from config."""
    model_fn = MODEL_REGISTRY[cfg["model"]]
    model = model_fn(num_classes=NUM_CLASSES, pretrained=cfg["pretrained"])
    return model.to(device)


def build_optimizer(model, cfg):
    """Build optimizer from config."""
    param_groups = get_optimizer_params(
        model,
        head_lr=cfg["head_lr"],
        backbone_lr=cfg["backbone_lr"],
        weight_decay=cfg["weight_decay"],
    )

    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    return optimizer


def build_scheduler(optimizer, cfg):
    """Build learning rate scheduler."""
    if cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    elif cfg["scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return None


def run_experiment(cfg, resume_path=None):
    """Run a single experiment."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Experiment: {cfg['name']} — {cfg['description']}")

    # Initialize WandB
    wandb.init(
        project="hw2-flower-recognition",
        name=cfg["name"],
        config=cfg,
    )

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_ROOT,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )

    # Model
    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    best_acc = 0.0

    # Resume from checkpoint
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"Resumed from {resume_path} (epoch {checkpoint['epoch']})")

    # Training loop
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"Epoch {epoch:2d}/{cfg['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = f"checkpoints/{cfg['name']}_best.pth"
            save_checkpoint(model, optimizer, epoch, best_acc, ckpt_path)
            wandb.run.summary["best_val_acc"] = best_acc

    # Final test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    wandb.run.summary["test_acc"] = test_acc
    print(f"\nFinal results for {cfg['name']}:")
    print(f"  Best Val Acc: {best_acc:.2f}%")
    print(f"  Test Acc:     {test_acc:.2f}%")

    wandb.finish()
    return best_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description="Task 1: Flower Recognition Training")
    parser.add_argument("--config", type=str, default="baseline_r18",
                        help="Experiment configuration name")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments sequentially")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.all:
        experiments = [
            "baseline_r18",
            "baseline_r34",
            "epochs_15",
            "epochs_50",
            "lr_high",
            "lr_low",
            "random_init",
            "se_resnet18",
            "cbam_resnet18",
        ]
        results = {}
        for exp_name in experiments:
            print(f"\n{'='*60}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*60}")
            cfg = get_config(exp_name)
            best_val, test_acc = run_experiment(cfg)
            results[exp_name] = {"best_val": best_val, "test": test_acc}

        print(f"\n{'='*60}")
        print("Summary of all experiments:")
        print(f"{'='*60}")
        for name, res in results.items():
            print(f"  {name:20s} | Best Val Acc: {res['best_val']:6.2f}% | Test Acc: {res['test']:6.2f}%")
    else:
        cfg = get_config(args.config)
        run_experiment(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
