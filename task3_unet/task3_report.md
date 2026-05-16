# Task 3: U-Net 语义分割实验报告

## 1. 任务目标

本任务要求从零搭建 U-Net 语义分割网络，并在 Stanford Background Dataset 上进行训练与验证。实验重点比较三种损失函数对分割性能的影响：

- Cross-Entropy Loss
- Dice Loss
- Cross-Entropy Loss + Dice Loss

主要评价指标为 pixel accuracy 和 mIoU，其中 mIoU 作为语义分割任务的核心对比指标。

## 2. 数据集与预处理

本实验使用 Stanford Background Dataset 的 `iccv09Data` 版本，数据目录为：

```text
data/iccv09Data/
  images/
  labels/
  horizons.txt
```

其中语义分割标签来自：

```text
labels/*.regions.txt
```

标签类别共 8 类：

| id | 类别 |
| --- | --- |
| 0 | sky |
| 1 | tree |
| 2 | road |
| 3 | grass |
| 4 | water |
| 5 | building |
| 6 | mountain |
| 7 | foreground object |

标签中的 `-1` 表示 unknown 区域，训练和评估时统一映射为 `ignore_index=255`，不参与 loss 与 metric 计算。

数据划分方式为随机划分：

```text
total = 715
train = 572
val = 143
val_ratio = 0.2
```

图像统一 resize 到：

```text
256 x 320
```

图像归一化使用 ImageNet 均值与方差：

```text
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

## 3. 模型结构

本实验实现了一个标准 U-Net 网络。整体结构为 4 层下采样 encoder、1 个 bottleneck 和 4 层上采样 decoder。

每个卷积块 `DoubleConv` 由以下结构组成：

```text
Conv 3x3 -> BatchNorm -> ReLU -> Conv 3x3 -> BatchNorm -> ReLU
```

下采样模块：

```text
MaxPool 2x2 -> DoubleConv
```

上采样模块：

```text
ConvTranspose2d -> Skip Connection concat -> DoubleConv
```

默认输入为 `[B, 3, 256, 320]`，各层输出如下：

| 阶段 | 输出尺寸 |
| --- | --- |
| input | `[B, 3, 256, 320]` |
| encoder 1 | `[B, 64, 256, 320]` |
| encoder 2 | `[B, 128, 128, 160]` |
| encoder 3 | `[B, 256, 64, 80]` |
| encoder 4 | `[B, 512, 32, 40]` |
| bottleneck | `[B, 1024, 16, 20]` |
| decoder 1 | `[B, 512, 32, 40]` |
| decoder 2 | `[B, 256, 64, 80]` |
| decoder 3 | `[B, 128, 128, 160]` |
| decoder 4 | `[B, 64, 256, 320]` |
| output logits | `[B, 8, 256, 320]` |

Skip Connection 通过在通道维度拼接 encoder 与 decoder 的同尺度特征实现，用于保留浅层空间细节，改善分割边界。

## 4. 损失函数

### 4.1 Cross-Entropy Loss

Cross-Entropy Loss 对每个有效像素进行多分类监督：

```text
L_ce = CE(logits, target)
```

其中标签为 `255` 的 unknown 像素被忽略。

### 4.2 Dice Loss

Dice Loss 首先对 logits 做 softmax，得到每个类别的概率，然后将标签转换为 one-hot 格式，逐类别计算 soft Dice：

```text
Dice = (2 * intersection + smooth) / (prediction + target + smooth)
L_dice = 1 - mean(Dice)
```

本实验中 `smooth=1.0`，用于避免分母为 0 并提升训练稳定性。

### 4.3 混合损失

混合损失直接将两者相加：

```text
L = L_ce + L_dice
```

Cross-Entropy Loss 更关注逐像素分类正确性，Dice Loss 更关注预测区域与真实区域的整体重叠程度。

## 5. 实验设置

主要训练超参数如下：

| 项目 | 设置 |
| --- | --- |
| framework | PyTorch |
| optimizer | AdamW |
| epochs | 100 |
| batch size | 8 |
| learning rate | 0.0002 |
| weight decay | 0.0001 |
| scheduler | CosineAnnealingLR |
| image size | 256 x 320 |
| base channels | 64 |
| num classes | 8 |
| mixed precision | enabled |
| logging | wandb + local CSV |

每个 epoch 记录：

```text
loss/train
loss/val
pixel_acc/train
pixel_acc/val
miou/train
miou/val
```

## 6. 实验结果

由于 `runs/ce/metrics.csv` 中包含多次追加训练记录，下面表格选择每种 loss 下当前日志中表现最好的完整实验结果。对于 CE，采用包含 train/val 指标的最新完整 100 epoch 记录。

| Loss | Best Epoch | Val Loss | Val Accuracy | Val mIoU | Train mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-Entropy | 67 | 0.5832 | 0.8239 | **0.6463** | 0.8796 |
| Dice | 84 | 0.3108 | 0.8185 | 0.6371 | 0.7622 |
| Cross-Entropy + Dice | 56 | 0.9532 | 0.8178 | 0.6341 | 0.8361 |

从 best mIoU 来看，三种损失函数均能达到约 0.63 以上的验证集 mIoU。其中 Cross-Entropy Loss 在当前实验中取得最高验证 mIoU，为 `0.6463`。Dice Loss 的结果略低，为 `0.6371`。混合损失为 `0.6341`，没有超过单独 CE。

各实验最后一个 epoch 的结果如下：

| Loss | Last Epoch | Val Loss | Val Accuracy | Val mIoU |
| --- | ---: | ---: | ---: | ---: |
| Cross-Entropy | 100 | 0.6243 | 0.8195 | 0.6358 |
| Dice | 100 | 0.3188 | 0.8157 | 0.6242 |
| Cross-Entropy + Dice | 100 | 1.0562 | 0.8149 | 0.6154 |

可以看到，三种设置在后期都出现一定程度的平台期或轻微回落。因此最终模型选择不应使用 last checkpoint，而应使用按验证集 mIoU 保存的 best checkpoint。

## 7. 结果分析

Cross-Entropy Loss 的验证 mIoU 最高，说明在该自然场景 8 类语义分割任务上，逐像素分类监督已经能够提供较强的优化信号。Stanford Background Dataset 的类别数量较少，且大面积类别如 sky、tree、building 等占比较高，因此 CE 能较稳定地学习主要语义区域。

Dice Loss 的 best mIoU 略低于 CE，但仍达到相近水平。Dice Loss 更关注区域重叠，对类别不均衡通常更友好；不过它基于 batch 统计类别区域，当 batch size 较小且部分类别缺失时，优化信号会更不稳定。因此在本实验中 Dice 并未明显超过 CE。

混合损失没有带来进一步提升，可能原因是当前采用简单的 `CE + Dice` 等权相加，两种 loss 的数值尺度和梯度贡献并不完全匹配。后续可以尝试：

```text
CE + 0.5 * Dice
CE + 0.3 * Dice
0.5 * CE + Dice
```

此外，当前训练主要使用 resize 和归一化，没有加入随机裁剪、水平翻转、颜色扰动等数据增强。对于 715 张图像的小数据集，数据增强很可能比继续微调 loss 权重带来更明显提升。

## 8. 可视化结果

本实验额外实现了单图可视化脚本 `visualize.py`，可输出原图、预测 mask、预测叠加图以及可选 GT 对比图。当前已有可视化结果：

| Loss | Visualization |
| --- | --- |
| Cross-Entropy | `runs/ce/vis_0000047_gt.png` |
| Dice | `runs/dice/vis_0000047_gt.png` |
| Cross-Entropy + Dice | `runs/ce_dice/vis_0000047_gt.png` |

可视化命令示例：

```bash
python visualize.py \
  --checkpoint checkpoints/ce/best.pt \
  --image data/iccv09Data/images/0000047.jpg \
  --gt data/iccv09Data/labels/0000047.regions.txt \
  --output runs/ce/vis_0000047_gt.png
```

## 9. 小结

本任务完成了从零搭建 U-Net、读取 Stanford Background Dataset、实现 CE/Dice/混合损失、训练验证与 mIoU 评估的完整流程。实验结果表明，在当前训练设置下 Cross-Entropy Loss 表现最好，best val mIoU 达到 `0.6463`。Dice Loss 与混合损失表现接近但略低，说明 Dice 对该任务并非必然提升，需要进一步调节 loss 权重、batch size 或配合数据增强。

后续优化方向包括：

- 加入随机水平翻转、随机缩放裁剪和颜色扰动。
- 调整混合损失中 CE 与 Dice 的权重。
- 记录 per-class IoU，分析具体哪些类别拖累 mIoU。
- 尝试较小的 `base_channels=32` 以缓解小数据集过拟合。
