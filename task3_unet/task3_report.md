# Task 3: U-Net 语义分割实验报告

## 1. 任务

本任务要求从零搭建 U-Net 语义分割网络，并在 Stanford Background Dataset 上进行训练与验证。实验核心目标包括：

- 实现 U-Net 的下采样、上采样与 Skip Connection 结构。
- 使用 Stanford Background Dataset 进行语义分割训练。
- 比较 Cross-Entropy Loss、Dice Loss 以及 Cross-Entropy + Dice 混合损失在 mIoU 上的表现。
- 记录并分析训练过程中的 loss、pixel accuracy 和 mIoU 曲线。
- 对不同损失函数得到的预测结果进行可视化 case study。

本实验主要评价指标为 pixel accuracy 和 mIoU。其中 pixel accuracy 衡量所有有效像素的整体分类正确率，mIoU 则先计算每个类别的 IoU 再取平均，更能反映语义分割任务中不同类别的整体分割质量。

## 2. 数据集与预处理

本实验使用 Stanford Background Dataset 的 `iccv09Data` 版本。该数据集包含 715 张自然场景图像，语义标签共 8 类：

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

标签文件中的 `-1` 表示 unknown 区域。实验中将其映射为 `ignore_index=255`，在 loss 计算和 metric 统计时均忽略，不参与训练监督与评估。

数据划分采用固定随机种子随机切分：

| Split | 数量 |
| --- | ---: |
| Train | 572 |
| Validation | 143 |
| Total | 715 |

预处理方式如下：

- 图像 resize 到 `256 x 320`。
- 标签使用 nearest neighbor resize，避免语义类别 id 被插值破坏。
- 图像使用 ImageNet 均值和方差归一化：

```text
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

图像与标签通过文件名 stem 配对。例如 `images/0000047.jpg` 与 `labels/0000047.regions.txt` 配对。

## 3. 模型结构与损失函数

### 3.1 U-Net 模型结构

本实验从零实现标准 U-Net。网络由 encoder、bottleneck 和 decoder 三部分组成：

- Encoder：4 次下采样，逐步降低空间分辨率并提升通道数。
- Bottleneck：位于网络最深层，提取高层语义特征。
- Decoder：4 次上采样，逐步恢复空间分辨率。
- Skip Connection：将 encoder 中同尺度的特征与 decoder 上采样后的特征在通道维度拼接，用于补充边界、纹理和位置信息。

基本卷积块 `DoubleConv` 结构为：

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

默认输入为 `[B, 3, 256, 320]`，各阶段输出尺寸如下：

| 阶段 | 输出尺寸 |
| --- | --- |
| Input | `[B, 3, 256, 320]` |
| Encoder 1 | `[B, 64, 256, 320]` |
| Encoder 2 | `[B, 128, 128, 160]` |
| Encoder 3 | `[B, 256, 64, 80]` |
| Encoder 4 | `[B, 512, 32, 40]` |
| Bottleneck | `[B, 1024, 16, 20]` |
| Decoder 1 | `[B, 512, 32, 40]` |
| Decoder 2 | `[B, 256, 64, 80]` |
| Decoder 3 | `[B, 128, 128, 160]` |
| Decoder 4 | `[B, 64, 256, 320]` |
| Output logits | `[B, 8, 256, 320]` |

最终输出为 logits，而不是 softmax 后的概率。训练时 Cross-Entropy Loss 直接接收 logits；计算 Dice Loss 时再对 logits 做 softmax。

### 3.2 Cross-Entropy Loss

Cross-Entropy Loss 对每个有效像素进行多分类监督：

```text
L_ce = CE(logits, target)
```

该损失直接优化逐像素分类正确性，训练稳定，是多分类语义分割中的常用基线。

### 3.3 Dice Loss

Dice Loss 首先对 logits 做 softmax 得到每个类别的预测概率，再将 target 转换为 one-hot 形式，逐类别计算 soft Dice：

```text
Dice = (2 * intersection + smooth) / (prediction + target + smooth)
L_dice = 1 - mean(Dice)
```

本实验使用 `smooth=1.0`，用于避免分母为 0 并提升训练稳定性。Dice Loss 更关注预测区域与真实区域的整体重叠程度，对类别不均衡场景通常更敏感。

### 3.4 混合损失

混合损失定义为：

```text
L = ce_weight * L_ce + dice_weight * L_dice
```

实验中比较了三组混合权重：

| 实验名 | ce_weight | dice_weight |
| --- | ---: | ---: |
| CE + Dice (0.5/0.5) | 0.5 | 0.5 |
| CE + Dice (0.2/0.8) | 0.2 | 0.8 |
| CE + Dice (0.8/0.2) | 0.8 | 0.2 |

通过调节权重，可以观察逐像素分类监督与区域重叠监督之间的平衡对最终 mIoU 的影响。

## 4. 实验设置

训练配置如下：

| 项目 | 设置 |
| --- | --- |
| Framework | PyTorch |
| Model | U-Net |
| Dataset | Stanford Background Dataset |
| Input size | 256 x 320 |
| Num classes | 8 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Epochs | 100 |
| Batch size | 8 |
| Learning rate | 0.0002 |
| Weight decay | 0.0001 |
| Mixed precision | Enabled |
| Logging | WandB + local CSV |

每个 epoch 记录以下指标：

```text
loss/train
loss/val
pixel_acc/train
pixel_acc/val
miou/train
miou/val
```

训练过程中根据验证集 mIoU 保存 best checkpoint，同时保存 last checkpoint。最终比较使用 best validation mIoU 对应的 epoch。

## 5. 实验结果与分析

### 5.1 Best Metrics

下表整理了各实验在验证集上取得 best mIoU 时的指标：

| Loss | Best Epoch | Val Loss | Val Acc | Val mIoU | Train mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-Entropy | 67 | 0.5832 | 0.8239 | **0.6463** | 0.8796 |
| Dice | 84 | 0.3108 | 0.8185 | 0.6371 | 0.7622 |
| CE + Dice (0.5/0.5) | 58 | 0.4828 | 0.8145 | 0.6340 | 0.8380 |
| CE + Dice (0.2/0.8) | 56 | 0.3981 | 0.8196 | 0.6377 | 0.8032 |
| CE + Dice (0.8/0.2) | 72 | 0.5764 | 0.8159 | 0.6318 | 0.8869 |

从 best mIoU 看，Cross-Entropy Loss 取得最高结果，为 `0.6463`。Dice Loss 和 CE + Dice (0.2/0.8) 表现接近，分别为 `0.6371` 和 `0.6377`。CE + Dice (0.8/0.2) 的训练集 mIoU 较高，但验证集 mIoU 较低，说明更偏向 CE 的混合权重可能带来更强拟合，但泛化收益有限。

### 5.2 Loss 曲线分析

![Loss Curves](runs/loss.png)

从 loss 曲线可以看到，所有实验的训练 loss 均持续下降，说明模型能够稳定收敛。验证 loss 在前期快速下降，中后期进入平台期，部分曲线出现轻微回升，表明继续训练后验证集收益有限，存在一定过拟合趋势。

需要注意的是，不同 loss 的数值尺度并不完全一致，因此不能直接用 loss 数值大小判断模型优劣。例如 Dice Loss 的数值范围与 Cross-Entropy Loss 不同，混合损失的大小也受 `ce_weight` 和 `dice_weight` 影响。更可靠的比较应基于验证集 mIoU 和 pixel accuracy。

### 5.3 mIoU 曲线分析

![mIoU Curves](runs/miou.png)

mIoU 曲线显示，各实验在前 20 到 40 个 epoch 提升较快，随后逐渐进入平台期。验证集 mIoU 最终集中在 `0.63` 到 `0.65` 左右，说明不同损失函数都能学习到较有效的语义分割表示。

Cross-Entropy 在最终 best mIoU 上略优，说明该数据集上逐像素分类监督已经足够强。Dice 和 CE + Dice 曲线整体接近，未显著超过 CE。可能原因包括：

- Stanford Background 是自然场景多分类分割，不是极端前景/背景不平衡任务。
- Dice Loss 基于 batch 内类别区域统计，batch size 为 8 时对小类别较敏感，曲线更容易受 batch 类别分布影响。
- 混合损失的权重仍需进一步搜索，简单线性组合不一定带来稳定提升。

### 5.4 Pixel Accuracy 曲线分析

![Pixel Accuracy Curves](runs/pixel_acc.png)

pixel accuracy 曲线整体比 mIoU 更平滑，并且各实验最终都达到约 `0.81` 到 `0.82` 的验证集 accuracy。由于 pixel accuracy 按所有有效像素整体统计，容易受到大面积类别影响。例如 sky、building、road 等区域面积较大，只要这些类别预测较好，accuracy 就会较高。

相比之下，mIoU 会对每个类别分别计算 IoU 后取平均，因此对小类别和类别边界更敏感。最终结果中不同方法的 pixel accuracy 差距较小，但 mIoU 仍能体现 CE、Dice 和混合损失之间的细微差异。

## 6. Case Study: 图像分割可视化

为了更直观地比较不同损失函数的预测行为，本实验随机选择图像利用训练的模型可视化语义分割结果。每张图从左到右依次为原图、预测 mask、预测叠加图、GT mask、GT 叠加图和类别图例。

### 6.1 Cross-Entropy Loss

![CE Case Study](runs/ce/vis_0000047_gt.png)

CE 模型能够较稳定地区分 sky、building、road 和 foreground object。预测结果整体较为保守，建筑与天空的大区域分割较清晰，前景人物也能被检测出来。但在人物边界和远处小物体区域仍存在边界不够精细的问题，部分 foreground 与 road 的交界区域存在混淆。

### 6.2 Dice Loss

![Dice Case Study](runs/dice/vis_0000047_gt.png)

Dice 模型对 foreground object 的响应更激进，能够覆盖主要人物区域，但也更容易把部分非前景区域误分为 foreground。例如右侧遮阳棚、路面或局部建筑附近出现较大面积的 foreground 误检。这与 Dice Loss 强调区域重叠有关：它可能倾向于扩大某些类别区域以提升覆盖，但会牺牲局部精确性。

### 6.3 Cross-Entropy + Dice Loss

![CE + Dice Case Study](runs/ce_dice/vis_0000047_gt.png)

混合损失的预测结果介于 CE 和 Dice 之间。它保留了 CE 对主要大区域的稳定分类能力，同时 foreground 区域比 CE 更活跃。但在道路与前景、建筑边界附近仍存在误分割，且没有在该样例中明显优于 CE。

### 6.4 Case Study 小结

从该样例可以观察到，三种方法都能较好恢复图像的主要语义布局：上方 sky、中部 building、下方 road 以及局部 foreground object。差异主要体现在 foreground object 和边界区域：

- CE 更保守，整体区域更稳定。
- Dice 更激进，foreground 覆盖更大，但误检也更多。
- CE + Dice 介于二者之间，但在当前权重下没有明显改善边界细节。

这与定量结果一致：Dice 与混合损失能够达到接近 CE 的 mIoU，但并未稳定超过 CE。

## 7. 总结

本任务完成了 U-Net 从零搭建、Stanford Background Dataset 加载、CE/Dice/CE+Dice 损失实现、训练验证、WandB 记录、指标分析和单图可视化。

实验结果表明：

- Cross-Entropy Loss 在当前设置下取得最高验证 mIoU，为 `0.6463`。
- Dice Loss 与 CE + Dice (0.2/0.8) 的验证 mIoU 接近，分别为 `0.6371` 和 `0.6377`。
- 混合损失并未必然提升 mIoU，说明 CE 与 Dice 的权重需要结合数据集特点进一步调节。
- Pixel accuracy 对大面积类别更敏感，而 mIoU 更能反映不同类别的平均分割质量，因此最终模型比较以 mIoU 为主。
- 可视化结果显示，Dice 更容易扩大 foreground 区域，CE 更保守稳定，混合损失介于二者之间。

后续可以从以下方向继续优化：

- 加入随机水平翻转、随机缩放裁剪和颜色扰动，提升小数据集泛化能力。
- 进一步搜索 `ce_weight` 和 `dice_weight`，例如 `0.1/0.9`、`0.3/0.7`、`0.7/0.3`。
- 统计 per-class IoU，分析 water、mountain、foreground 等小类别是否限制整体 mIoU。
- 尝试较小的 `base_channels=32` 或更强正则化，以缓解训练集 mIoU 明显高于验证集的现象。
