from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import functional as F
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXTENSIONS = IMAGE_EXTENSIONS | {".txt"}


def _find_first_existing(root: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find any of {candidates} under {root}")


class StanfordBackgroundDataset(Dataset):
    """Stanford Background Dataset loader for semantic segmentation.

    Expected layout:
        root/images/*.jpg
        root/labels/*.png

    The loader also accepts common folder aliases such as imgs, image, masks,
    annotations, and label.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (256, 320),
        ignore_index: int = 255,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = tuple(image_size)
        self.ignore_index = ignore_index
        self.image_transform = image_transform

        image_dir = _find_first_existing(self.root, ["images", "imgs", "image", "JPEGImages"])
        label_dir = _find_first_existing(
            self.root,
            ["labels", "label", "masks", "mask", "annotations", "SegmentationClass"],
        )

        self.images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        region_labels = sorted(label_dir.glob("*.regions.txt"))
        if region_labels:
            label_files = region_labels
        else:
            label_files = sorted(p for p in label_dir.iterdir() if p.suffix.lower() in LABEL_EXTENSIONS)
        labels_by_stem = {}
        for path in label_files:
            stem = path.stem.removesuffix(".regions")
            labels_by_stem[stem] = path

        self.samples: list[tuple[Path, Path]] = []
        for image_path in self.images:
            label_path = labels_by_stem.get(image_path.stem)
            if label_path is not None:
                self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(
                f"No image/label pairs found in {image_dir} and {label_dir}. "
                "Please make sure paired files share the same stem."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)

        if self.image_transform is not None:
            image_tensor = self.image_transform(image)
        else:
            image_tensor = TF.to_tensor(image)
            image_tensor = TF.normalize(
                image_tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )

        if label_path.suffix.lower() == ".txt":
            label_array = np.loadtxt(label_path, dtype=np.int64)
            label_array[label_array < 0] = self.ignore_index
            label_tensor = torch.from_numpy(label_array).long()
            label_tensor = F.interpolate(
                label_tensor[None, None].float(),
                size=self.image_size,
                mode="nearest",
            )[0, 0].long()
            return image_tensor, label_tensor
        else:
            label = Image.open(label_path)

        label = TF.resize(label, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        label_array = np.array(label, dtype=np.int64)

        # Some annotations are RGB color maps. This converts each unique color to
        # a stable class id so training can still start without a manual palette.
        if label_array.ndim == 3:
            h, w, _ = label_array.shape
            colors, inverse = np.unique(label_array.reshape(-1, 3), axis=0, return_inverse=True)
            label_array = inverse.reshape(h, w).astype(np.int64)

        label_tensor = torch.from_numpy(label_array).long()
        label_tensor[label_tensor < 0] = self.ignore_index
        return image_tensor, label_tensor


def build_dataloaders(
    root: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    seed: int,
    ignore_index: int = 255,
) -> tuple[DataLoader, DataLoader]:
    dataset = StanfordBackgroundDataset(
        root=root,
        image_size=image_size,
        ignore_index=ignore_index,
    )

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    print(
        f"Dataset split: total={len(dataset)}, train={len(train_set)}, "
        f"val={len(val_set)}, val_ratio={val_ratio}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
