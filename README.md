# 项目名称

> 一句话简介：基于 SAMed 的医学图像分割复现与改进（示例，请修改为你的描述）。

## 项目简介

（用 2-3 句话说明：研究什么问题、用了什么方法、达到什么效果。例如：本仓库基于 CVPR 2023 的 SAMed 工作，使用 LoRA 微调 Segment Anything Model（SAM）完成医学图像分割，并针对 XX 问题进行了改进，在 XX 数据集上 Dice 达到 XX。）

## 方法要点

- 基线：SAMed（LoRA 微调 SAM，CVPR 2023）
- 改进点 1：（你的改动，例如：引入 XX 模块 / 调整损失函数）
- 改进点 2：（你的改动）

## 环境依赖

- Python 3.x / PyTorch 2.x / CUDA 11.x
- 安装依赖：`pip install -r requirements.txt`

## 快速开始

### 训练

```bash
# 按你的实际训练命令填写
python train.py --config xxx
```

### 推理 / 界面

```bash
python mainUI.py
```

## 结果

（放分割效果对比图、训练曲线等，一图胜千言）

## 目录结构

（简要说明，例如：）

```
├── mainUI.py                # 主程序入口
├── sam_lora_image_encoder.py# LoRA 微调 SAM 的编码器实现
├── segment_anything/        # SAM 基础组件
├── checkpoints/             # 模型权重（本地，不入库）
└── train_npz/               # 训练数据（本地，不入库）
```

## 引用

- Zhang K. et al. SAMed: Segment Anything Model Meets Medical Image Segmentation. CVPR 2023.
- 本仓库基于该工作复现并改进，具体改进见"方法要点"。

## License

（可选：MIT / 学术用途声明。若毕设归属学校或导师，请先确认后填写。）
