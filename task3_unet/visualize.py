import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from models import UNet
from utils import get_device


CLASS_NAMES = [
    "sky",
    "tree",
    "road",
    "grass",
    "water",
    "building",
    "mountain",
    "foreground",
]

PALETTE = np.array(
    [
        [128, 192, 255],
        [0, 128, 0],
        [128, 128, 128],
        [64, 192, 64],
        [0, 128, 192],
        [192, 128, 64],
        [128, 96, 64],
        [192, 64, 192],
    ],
    dtype=np.uint8,
)


def load_image(path: str | Path, image_size: tuple[int, int]) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    resized = TF.resize(image, image_size, interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return resized, tensor.unsqueeze(0)


def load_label(path: str | Path, image_size: tuple[int, int], ignore_index: int) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".txt":
        label = np.loadtxt(path, dtype=np.int64)
        label[label < 0] = ignore_index
        label_tensor = torch.from_numpy(label)[None, None].float()
        label_tensor = torch.nn.functional.interpolate(
            label_tensor,
            size=image_size,
            mode="nearest",
        )[0, 0].long()
        return label_tensor.numpy()

    label_image = Image.open(path)
    label_image = TF.resize(label_image, image_size, interpolation=TF.InterpolationMode.NEAREST)
    label = np.array(label_image, dtype=np.int64)
    label[label < 0] = ignore_index
    return label


def colorize_mask(mask: np.ndarray, ignore_index: int) -> Image.Image:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ignore_index
    class_ids = mask[valid] % len(PALETTE)
    color[valid] = PALETTE[class_ids]
    color[~valid] = np.array([0, 0, 0], dtype=np.uint8)
    return Image.fromarray(color)


def overlay_mask(image: Image.Image, mask: np.ndarray, ignore_index: int, alpha: float = 0.45) -> Image.Image:
    color_mask = colorize_mask(mask, ignore_index).convert("RGB")
    return Image.blend(image.convert("RGB"), color_mask, alpha=alpha)


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_height = 30
    canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    return canvas


def make_legend() -> Image.Image:
    row_height = 24
    width = 220
    height = row_height * len(CLASS_NAMES)
    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for idx, name in enumerate(CLASS_NAMES):
        y = idx * row_height
        draw.rectangle((8, y + 5, 26, y + 20), fill=tuple(PALETTE[idx].tolist()))
        draw.text((34, y + 4), f"{idx}: {name}", fill=(0, 0, 0), font=font)
    return add_title(legend, "Legend")


def concat_h(images: list[Image.Image]) -> Image.Image:
    height = max(image.height for image in images)
    width = sum(image.width for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize U-Net semantic segmentation results.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--gt", default=None, help="Optional ground-truth label path.")
    parser.add_argument("--output", default="visualization.png", help="Output visualization image.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    image_size = tuple(cfg["data"]["image_size"])
    ignore_index = cfg["data"]["ignore_index"]

    device = get_device(args.device)
    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["data"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image, image_tensor = load_image(args.image, image_size)
    logits = model(image_tensor.to(device))
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)

    panels = [
        add_title(image, "Image"),
        add_title(colorize_mask(pred, ignore_index), "Prediction Mask"),
        add_title(overlay_mask(image, pred, ignore_index), "Prediction Overlay"),
    ]

    if args.gt is not None:
        gt = load_label(args.gt, image_size, ignore_index)
        panels.extend(
            [
                add_title(colorize_mask(gt, ignore_index), "GT Mask"),
                add_title(overlay_mask(image, gt, ignore_index), "GT Overlay"),
            ]
        )

    panels.append(make_legend())
    output = concat_h(panels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
