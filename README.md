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
│   ├── results.md                     # 实验结果报告
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
└── task3_unet/                        # 任务3：U-Net 语义分割（待完成）
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
- 102 类花卉，共 8189 张图片
- 自动下载：首次运行训练脚本时自动下载

### 运行实验

```bash
cd task1_flower

# 运行单个实验
python train.py --config baseline_r18       # ResNet-18 基线
python train.py --config baseline_r34       # ResNet-34 深度对比
python train.py --config random_init        # 随机初始化（预训练消融）
python train.py --config se_resnet18        # SE 注意力机制
python train.py --config cbam_resnet18      # CBAM 注意力机制

# 超参数分析
python train.py --config epochs_15          # 15 epochs
python train.py --config epochs_50          # 50 epochs
python train.py --config lr_high            # 高学习率
python train.py --config lr_low             # 低学习率

# 运行所有实验
python train.py --all
```

### 实验结果汇总

| 排名 | 模型 | 参数量 | Test Acc |
|------|------|--------|----------|
| 🥇 | ResNet-34 (预训练) | 21.3M | **97.88%** |
| 🥈 | SE-ResNet-18 | 11.3M | **97.64%** |
| 🥉 | ResNet-18 (预训练) | 11.2M | 97.23% |
| 4 | CBAM-ResNet-18 | 11.3M | 95.12% |
| 5 | ResNet-18 (随机初始化) | 11.2M | 75.35% |

详细报告见 [task1_flower/results.md](task1_flower/results.md)

### 关键结论

1. **预训练权重至关重要**：从零训练 vs 预训练微调差距达 22 个百分点
2. **SE Block 有效**：通道注意力提升 +0.41%，CBAM 反而降低 -2.11%
3. **网络深度**：ResNet-34 (97.88%) 优于 ResNet-18 (97.23%)
4. **无过拟合**：所有预训练模型训练/验证差距小

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

## 任务3: U-Net 从零搭建与损失函数工程（TODO）

---

## WandB 监控

所有实验使用 WandB 记录训练指标。查看曲线请在项目目录下找到 wandb 运行目录：

```bash
# 查看本地 wandb 数据
wandb sync wandb/
```

## 模型权重

训练好的 `.pth` 权重文件保存在 `task1_flower/checkpoints/` 目录下。
