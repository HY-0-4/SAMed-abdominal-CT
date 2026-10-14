# 基于 SAMed 的腹部多器官 CT 图像分割系统

> 基于 Segment Anything Model (SAM) 的医学图像分割复现与改进，支持 8 种腹部器官的自动分割，并提供可视化交互界面。

## 项目简介

本项目基于 CVPR 2023 的 SAMed 工作，使用 LoRA（Low-Rank Adaptation）微调 Segment Anything Model（SAM）的图像编码器，实现腹部 CT 图像中多器官的自动分割。在此基础上，引入了测试时增强（Test-Time Augmentation, TTA）策略，并实现了基于不确定性加权的多视角融合方法，进一步提升分割结果的稳定性与精度。配套开发了基于 PyQt5 的可视化交互系统，支持 NIfTI 医学影像浏览、窗宽窗位调节、分割结果叠加显示与多病例对比。

## 方法要点

- **基线模型**：SAMed —— LoRA 微调 SAM ViT-B，仅训练低秩适配矩阵，冻结原始模型参数，大幅降低微调成本
- **改进点 1：测试时增强（TTA）** —— 推理阶段对输入进行多种变换（翻转、旋转等），多次预测后融合，降低单次预测的偶然性
- **改进点 2：不确定性加权融合** —— 基于预测熵估计各次增强结果的置信度，对高置信度区域赋予更高权重，相比简单平均进一步提升边界分割精度
- **可视化系统**：PyQt5 桌面端应用，支持 NIfTI 文件加载、切片浏览、器官分割叠加、指标实时展示

## 分割器官

| 编号 | 器官 | 颜色标识 |
|------|------|----------|
| 1 | 脾脏 | 红色 |
| 2 | 右肾 | 绿色 |
| 3 | 左肾 | 青色 |
| 4 | 胆囊 | 黄色 |
| 5 | 肝脏 | 橙色 |
| 6 | 胃 | 紫色 |
| 7 | 主动脉 | 蓝色 |
| 8 | 胰腺 | 品红 |

## 环境依赖

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 主要依赖：
  ```bash
  pip install torch torchvision
  pip install pyqt5 nibabel safetensors
  pip install opencv-python matplotlib
  ```

## 快速开始

### 运行可视化界面

```bash
python mainUI.py
```

界面功能包括：
- 左侧病例列表选择不同 CT 病例
- 滚轮或滑块浏览不同切片
- 调节窗宽窗位适配不同组织
- 切换显示原始 CT / 分割结果叠加 / 金标准对比
- 查看各器官的 DSC、HD95 等分割指标
- 对比普通推理与 TTA 增强后的分割效果

### 模型训练

```bash
# 基于 LoRA 的 SAM 微调训练
python train.py --data_root ./train_npz --lora_r 4 --epochs 100
```

## 结果展示

（建议在这里放几张分割效果图，比如：
- 一张腹部 CT 原始图 + 分割叠加的对比图
- 一张 TTA 前后效果对比图
- 一张界面截图）

### 量化指标（示例）

| 器官 | DSC (%) | HD95 (mm) |
|------|---------|-----------|
| 脾脏 | — | — |
| 右肾 | — | — |
| 左肾 | — | — |
| 胆囊 | — | — |
| 肝脏 | — | — |
| 胃 | — | — |
| 主动脉 | — | — |
| 胰腺 | — | — |

> 注：具体指标数值请替换为你实验的实际结果。

## 目录结构

```
├── mainUI.py                 # PyQt5 可视化交互系统主程序
├── sam_lora_image_encoder.py # LoRA 微调 SAM 的编码器实现
├── segment_anything/         # SAM 基础模型组件
│   ├── modeling/             # 模型结构（image encoder、mask decoder 等）
│   ├── utils/                # 工具函数
│   └── predict.py            # SAM Predictor
├── checkpoints/              # 模型权重（本地，不入库）
├── train_npz/                # 训练数据（本地，不入库）
└── output/                   # 预测结果与日志（本地，不入库）
```

## 技术亮点

1. **参数高效微调**：LoRA 仅训练约 0.5% 的参数即可完成医学域适配，显存占用低
2. **测试时增强**：推理阶段多视角预测 + 不确定性加权融合，无需重训模型即可提升精度
3. **完整工程落地**：从模型微调 → 推理 → 可视化界面，形成完整的医学影像分割工具链

## 引用

- Zhang K, Zhou R, et al. *SAMed: Segment Anything Model Meets Medical Image Segmentation*. CVPR 2023.
- The original SAM paper: Kirillov A, et al. *Segment Anything*. ICCV 2023.

## License

本项目仅供学术研究与学习使用。
