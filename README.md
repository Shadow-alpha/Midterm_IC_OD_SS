# HW2: 深度学习与空间智能 — 计算机视觉期中作业

## 项目结构

```
hw2/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包
├── 作业要求.md                         # 作业任务说明书
│
├── task1_flower/                      # 任务1：花卉识别
│   ├── train.py                       # 统一训练入口
│   ├── configs.py                     # 实验配置
│   ├── task1_report.md                # 实验结果报告
│   ├── task1_wandb.png                # Wandb 训练曲线
│   ├── results_baseline_r18.png       # Baseline Loss/Accuracy 曲线
│   ├── comparison_all.png             # 所有实验对比曲线
│   ├── models/
│   │   ├── resnet_custom.py           # ResNet-18/34 输出层修改
│   │   ├── se_resnet.py               # ResNet + SE Block
│   │   └── cbam_resnet.py             # ResNet + CBAM
│   └── utils/
│       ├── dataset.py                 # 102 Flower 数据集加载
│       └── train_utils.py             # 训练/验证循环
│
├── task2_car/                         # 任务2：目标检测与跟踪
│   ├── report.md                      # 任务2实验报告
│   ├── config.yaml                    # 训练/跟踪配置
│   ├── train_yolov8.py                # 训练脚本
│   ├── track_and_count.py             # 跟踪与越线计数
│   ├── export_frames.py               # 导出指定帧
│   ├── download_dataset.py            # 数据集下载
│   └── utils.py                       # 公共工具函数
└── task3_unet/                        # 任务3：U-Net 语义分割
    ├── train.py                       # 训练入口，支持 CE/Dice/CE+Dice
    ├── eval.py                        # checkpoint 评估脚本
    ├── visualize.py                   # 单图预测与 GT 可视化
    ├── task3_report.md                # 任务3实验报告
    ├── configs/default.yaml           # 训练配置
    ├── models/unet.py                 # 从零实现 U-Net
    ├── datasets/stanford_background.py # Stanford Background 数据集加载
    ├── losses.py                      # Cross-Entropy、Dice、混合损失
    ├── metrics.py                     # Pixel Accuracy 与 mIoU
    ├── runs/                          # 训练日志与可视化结果
    └── checkpoints/                   # best/last 模型权重
```

## 环境配置

```bash
# 推荐使用 conda 创建环境
conda create -n hw2 python=3.10
conda activate hw2

# 安装依赖
pip install -r requirements.txt
```

### 依赖包

- torch>=2.0.0
- torchvision>=0.15.0
- wandb
- numpy
- matplotlib
- pillow
- scipy
- tqdm
- opencv-python

## 任务1: 卷积神经网络微调与花卉识别

### 数据集

[102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- 102 类花卉，共 8189 张图片（每类 40-258 张）
- 数据集划分：训练集 70%（5,732 张） / 验证集 15%（1,228 张） / 测试集 15%（1,229 张）
- 自动下载：首次运行训练脚本时自动下载并解压

### 实验设置

- **框架**：PyTorch 2.11.0 + torchvision 0.26.0
- **硬件**：MacBook Air (M3) + Apple MPS 加速
- **优化器**：AdamW (weight_decay=1e-4)
- **学习率调度**：CosineAnnealingLR（余弦退火）
- **图像尺寸**：224×224（ImageNet 归一化）
- **数据增强**：RandomResizedCrop、RandomHorizontalFlip、ColorJitter
- **损失函数**：CrossEntropyLoss
- **评价指标**：Top-1 Accuracy

### 网络结构

- **Baseline**：ResNet-18 (11.2M) / ResNet-34 (21.3M)，加载 ImageNet 预训练权重
- **改进模型**：
  - SE-ResNet-18：在 BasicBlock 后插入 Squeeze-and-Excitation 通道注意力模块
  - CBAM-ResNet-18：在 BasicBlock 后插入 CBAM（通道 + 空间注意力）模块
- **消融实验**：随机初始化 ResNet-18（从零训练，无预训练）

### 运行实验

```bash
cd task1_flower

# 运行单个实验
python train.py --config baseline_r18       # ResNet-18 基线
python train.py --config baseline_r34       # ResNet-34 深度对比
python train.py --config random_init        # 随机初始化（预训练消融）
python train.py --config se_resnet18        # SE 注意力机制
python train.py --config cbam_resnet18      # CBAM 注意力机制

# 超参数分析（可选）
python train.py --config epochs_15          # 15 epochs
python train.py --config epochs_50          # 50 epochs
python train.py --config lr_high            # 高学习率
python train.py --config lr_low             # 低学习率

# 运行所有实验
python train.py --all
```

### 实验结果汇总

| 排名 | 模型 | 参数量 | Test Acc | 训练速度 |
|------|------|--------|----------|----------|
| 🥇 | ResNet-34 (预训练) | 21.3M | **97.88%** | 1.5 it/s |
| 🥈 | SE-ResNet-18 | 11.3M | **97.64%** | 2.1 it/s |
| 🥉 | ResNet-18 (预训练) | 11.2M | 97.23% | 2.3 it/s |
| 4 | CBAM-ResNet-18 | 11.3M | 95.12% | 2.0 it/s |
| 5 | ResNet-18 (随机初始化) | 11.2M | 75.35% | 2.7 it/s |

### Wandb 训练可视化

![Wandb 训练曲线](task1_flower/task1_wandb.png)

包含 val_loss、val_acc、train_loss、train_acc 四张图表，展示 5 个模型的训练过程对比。

### 模型权重下载

所有训练好的模型权重已上传至 Google Drive：

**[Task 1 Model Checkpoints](https://drive.google.com/drive/folders/1XF57BAXe9MUAzNUgudhMIID1v_0j_BVx?usp=drive_link)**

包含以下模型权重：
- `baseline_r18_best.pth` - ResNet-18 基线模型（Test Acc: 97.23%）
- `baseline_r34_best.pth` - ResNet-34 基线模型（Test Acc: 97.88%）
- `se_resnet18_best.pth` - SE-ResNet-18 模型（Test Acc: 97.64%）
- `cbam_resnet18_best.pth` - CBAM-ResNet-18 模型（Test Acc: 95.12%）
- `random_init_best.pth` - 随机初始化 ResNet-18（Test Acc: 75.35%）

详细报告见 [task1_flower/task1_report.md](task1_flower/task1_report.md)

### 关键结论

1. **预训练权重至关重要**：预训练微调 vs 随机初始化差距达 **21.88%**（97.23% vs 75.35%）
2. **SE 注意力有效**：通道注意力提升 **+0.41%**，几乎零额外计算成本
3. **CBAM 适得其反**：空间注意力降低 **-2.11%**，花卉分类更依赖通道特征而非空间位置
4. **网络深度**：ResNet-34 (97.88%) 优于 ResNet-18 (97.23%)，但参数量翻倍、速度慢 30%
5. **快速收敛**：预训练模型 Epoch 1 即可达到 80%+ 验证准确率
6. **无过拟合**：所有预训练模型训练/验证差距均在 2% 以内

---

## 任务2: 场景目标检测与视频多目标跟踪

本任务在 [task2_car/](task2_car/) 目录下完成，使用 YOLOv8 进行车辆检测与跟踪，并实现越线计数与遮挡分析。

### 数据集

- Road Vehicle Images Dataset（Kaggle）
- 数据组织（YOLO 格式）：

```
<dataset_dir>/
	images/
		train/
		val/
	labels/
		train/
		val/
```

### 训练与推理

```bash
cd task2_car

# 训练
python train_yolov8.py --config config.yaml

# 跟踪 + 越线计数
python track_and_count.py --config config.yaml
```

### 主要功能

- YOLOv8 车辆检测 + ByteTrack 跟踪（稳定 ID）
- 虚拟线越线计数（基于中心点符号变化）
- 遮挡片段帧导出（用于 ID 跳变分析）

### 结果与说明

- 训练与验证曲线、PR 曲线、混淆矩阵等结果见 [task2_car/report.md](task2_car/report.md)

## 任务3: U-Net 从零搭建与损失函数工程

本任务在 [task3_unet/](task3_unet/) 目录下完成，从零实现 U-Net 语义分割网络，并在 Stanford Background Dataset 上比较 Cross-Entropy Loss、Dice Loss 和 Cross-Entropy + Dice Loss 的 mIoU 表现。

### 数据集

- Stanford Background Dataset (`iccv09Data`)
- 共 715 张图像，语义标签来自 `labels/*.regions.txt`
- 8 个语义类别：sky、tree、road、grass、water、building、mountain、foreground object
- `-1` unknown 标签映射为 `ignore_index=255`，不参与训练和评估
- 数据划分：train 572 / val 143

数据目录：

```
task3_unet/data/iccv09Data/
	images/
	labels/
	horizons.txt
```

### 模型与训练

- 模型：4 层 encoder + bottleneck + 4 层 decoder 的 U-Net
- Skip Connection：decoder 每层与 encoder 同尺度特征在通道维度拼接
- 输入尺寸：256 x 320
- Optimizer：AdamW
- Scheduler：CosineAnnealingLR
- Epochs：100
- Batch size：8
- Learning rate：2e-4
- Logging：WandB + local CSV

运行三组损失函数实验：

```bash
cd task3_unet

python train.py --loss ce
python train.py --loss dice
python train.py --loss ce_dice
```

评估 best checkpoint：

```bash
python eval.py --checkpoint checkpoints/ce/best.pt
python eval.py --checkpoint checkpoints/dice/best.pt
python eval.py --checkpoint checkpoints/ce_dice/best.pt
```

单图可视化：

```bash
python visualize.py \
  --checkpoint checkpoints/ce/best.pt \
  --image data/iccv09Data/images/0000047.jpg \
  --gt data/iccv09Data/labels/0000047.regions.txt \
  --output runs/ce/vis_0000047_gt.png
```

### 实验结果汇总

| Loss | Best Epoch | Val Loss | Val Acc | Val mIoU |
|------|-----------:|---------:|--------:|---------:|
| Cross-Entropy | 67 | 0.5832 | 0.8239 | **0.6463** |
| Dice | 84 | 0.3108 | 0.8185 | 0.6371 |
| Cross-Entropy + Dice | 56 | 0.9532 | 0.8178 | 0.6341 |

当前实验中 Cross-Entropy Loss 取得最高验证集 mIoU。Dice Loss 与混合损失结果接近，但未超过 CE，可能与 batch size 较小、类别分布不均衡以及混合损失权重未调优有关。

详细报告见 [task3_unet/task3_report.md](task3_unet/task3_report.md)

---

## WandB 监控

所有实验使用 WandB 记录训练指标。查看曲线请在项目目录下找到 wandb 运行目录：

```bash
# 查看本地 wandb 数据
wandb sync wandb/
```

## 模型权重

训练好的 `.pth` 权重文件保存在 `task1_flower/checkpoints/` 目录下。
