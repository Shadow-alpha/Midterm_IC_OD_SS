"""
102 Category Flower Dataset loader.

Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
Images: 102flowers.tgz
Labels: imagelabels.mat

The dataset contains 102 classes with ~40-258 images per class.
Total: ~8189 images.
Splits: 70% train, 15% val, 15% test (random).
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import scipy.io
import tarfile
import urllib.request
from pathlib import Path
import numpy as np


FLOWER_URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
    "labels": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Flower102Dataset(Dataset):
    """102 Category Flower Dataset."""

    def __init__(self, root, split="train", transform=None, seed=42):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Download images and labels if needed
        self._download()

        # Load all labels (1-indexed, convert to 0-indexed)
        labels_mat = scipy.io.loadmat(self.root / "imagelabels.mat")
        all_labels = labels_mat["labels"][0] - 1  # shape: (8189,)

        # Build sorted image paths
        all_paths = [self.root / "jpg" / f"image_{i+1:05d}.jpg" for i in range(len(all_labels))]

        # Randomly shuffle and split 70/15/15
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(all_labels))
        n = len(perm)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        selected_idx = {
            "train": perm[:n_train],
            "val": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:],
        }[split]

        self.image_paths = [all_paths[i] for i in selected_idx]
        self.labels = [all_labels[i] for i in selected_idx]

    def _download(self):
        """Download dataset if not exists."""
        if (self.root / "jpg").exists():
            return

        print(f"Downloading 102 Flower dataset to {self.root}...")
        self.root.mkdir(parents=True, exist_ok=True)

        tar_path = self.root / "102flowers.tgz"
        if not tar_path.exists():
            urllib.request.urlretrieve(FLOWER_URLS["images"], tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=self.root)

        if not (self.root / "imagelabels.mat").exists():
            urllib.request.urlretrieve(FLOWER_URLS["labels"], self.root / "imagelabels.mat")

        tar_path.unlink()
        print("Download complete.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    """Get data transforms for train/val/test."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def get_dataloaders(root, batch_size=32, num_workers=4):
    """Create train/val/test dataloaders."""
    train_transform, val_transform = get_transforms()

    train_dataset = Flower102Dataset(root, split="train", transform=train_transform)
    val_dataset = Flower102Dataset(root, split="val", transform=val_transform)
    test_dataset = Flower102Dataset(root, split="test", transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
