# Task 3: U-Net Semantic Segmentation

This folder implements Task 3 from HW2: build U-Net from scratch and compare
Cross-Entropy Loss, Dice Loss, and Cross-Entropy + Dice Loss on the Stanford
Background Dataset.

## Project Structure

```text
task3_unet/
  configs/default.yaml
  datasets/stanford_background.py
  models/unet.py
  losses.py
  metrics.py
  train.py
  eval.py
  requirements.txt
```

## Dataset

Put the Stanford Background Dataset under:

```text
task3_unet/data/iccv09Data/
  images/
    xxx.jpg
  labels/
    xxx.regions.txt
```

The loader also accepts common aliases such as `imgs`, `masks`, `annotations`,
and `SegmentationClass`. For the official ICCV09 Stanford Background release,
semantic labels are loaded from `labels/*.regions.txt`; `-1` unknown pixels are
mapped to `ignore_index=255`.

The default number of semantic classes is `8`, matching the common Stanford
Background setup. If your processed labels use a different class count, update
`data.num_classes` in `configs/default.yaml`.

## Environment

```bash
cd task3_unet
pip install -r requirements.txt
```

If you do not want to use Weights & Biases, keep `logging.use_wandb: false` in
the config. Training metrics are always saved locally as CSV files.

## Train

Run the three required loss settings:

```bash
python train.py --loss ce
python train.py --loss dice
python train.py --loss ce_dice
```

Useful overrides:

```bash
python train.py --data-root data/iccv09Data --epochs 80 --batch-size 8 --lr 1e-4 --loss ce_dice
```

For mixed CE + Dice loss, override the two weights from command line:

```bash
python train.py --loss ce_dice --ce-weight 1.0 --dice-weight 0.5
python train.py --loss ce_dice --ce-weight 0.5 --dice-weight 1.0
```

Mixed-loss checkpoints and logs include the weights in their folder/run names,
for example:

```text
checkpoints/ce_dice_ce1_dice0p5/best.pt
runs/ce_dice_ce1_dice0p5/metrics.csv
wandb run name: unet-ce_dice_ce1_dice0p5
```

Default hyperparameters:

```text
batch size: 8
learning rate: 1e-4
optimizer: AdamW
epochs: 50
image size: 256 x 320
loss function: ce / dice / ce_dice
ce_weight: 0.5
dice_weight: 0.5
metric: pixel accuracy, mIoU
```

Checkpoints are saved to:

```text
checkpoints/{loss}/best.pt
checkpoints/{loss}/last.pt
```

CSV logs are saved to:

```text
runs/{loss}/metrics.csv
```

When Weights & Biases is enabled, metrics are logged with grouped names such as
`loss/train`, `loss/val`, `miou/train`, and `miou/val`, so train/validation
curves can be shown together in the same W&B panel.

## Evaluate

```bash
python eval.py --checkpoint checkpoints/ce/best.pt
python eval.py --checkpoint checkpoints/dice/best.pt
python eval.py --checkpoint checkpoints/ce_dice/best.pt
```

## Visualize

Visualize one image with a trained checkpoint:

```bash
python visualize.py --checkpoint checkpoints/ce/best.pt --image data/iccv09Data/images/0000047.jpg --output runs/vis_0000047.png
```

If ground truth is available, pass the corresponding `*.regions.txt` file:

```bash
python visualize.py --checkpoint checkpoints/ce/best.pt --image data/iccv09Data/images/0000047.jpg --gt data/iccv09Data/labels/0000047.regions.txt --output runs/vis_0000047_gt.png
```

## Report Template

Use the final rows from `runs/{loss}/metrics.csv` or evaluate each best
checkpoint and summarize:

| Loss | Pixel Accuracy | mIoU |
| --- | ---: | ---: |
| Cross-Entropy | TBD | TBD |
| Dice | TBD | TBD |
| Cross-Entropy + Dice | TBD | TBD |

For the PDF report, include the dataset split, batch size, learning rate,
optimizer, epoch count, loss curves, and the mIoU comparison table.
