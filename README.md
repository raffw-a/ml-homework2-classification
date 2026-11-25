# ImageNet 分类实验

本仓库用于课程分类作业。目标是统一数据划分、模型调用和实验结果格式，便于 A、B、C、D 并行协作。

## 实验顺序

1. **确认数据访问方式与范围（全组）**：确认 ImageNet 子集/版本、访问授权、数据托管位置及能否按分片读取。ImageNet 很大，且官方数据访问受条款约束；不要把完整数据集提交到 Git。若使用全量 ImageNet-1K，先评估算力和时间；SVM、决策树建议在预先固定的类别和样本子集上比较，CNN 与其他模型必须使用同一划分做公平比较。
2. **A：数据清单与划分**：确定类别映射、样本 ID、固定随机种子和 `train/val/test` 清单；训练、验证、测试样本互斥。只从训练部分拟合预处理参数。优先采用远端分片/流式数据源；若数据源不支持流式，就由一台获授权的共享机器保存一份数据，其他成员不必各自下载。
3. **B：模型实现与基线**：在 `model/` 新建模型文件，实现统一的 `fit(...)`、`predict(...)` 接口。基础比较至少包含两种不同传统算法；当前候选是 SVM 和决策树，CNN 属于深度学习拓展。图像批次接口与传统模型展平矩阵接口见 `train/contracts.py`。
4. **D：超参数分析**：只根据验证集选择参数，记录每个实验的配置和验证指标；至少比较一种模型调参前后的结果。测试集不能用于调参或特征选择。
5. **冻结实验配置**：确定最终模型、参数、预处理和种子后再评测测试集。全部模型使用相同测试样本和指标口径。
6. **C：可视化与分析**：读取 `runs/<run_id>/metrics.json` 和 `predictions.csv`，绘制模型性能对比、真实值与预测值/混淆矩阵，并解释实验现象。
7. **整理报告与复现说明**：记录数据来源及授权/获取方式、数据范围、划分方法、运行环境、依赖、命令、分工和结果分析。检查仓库不含数据集文件、缓存、密钥或模型权重。

## 目录结构

```text
.
├── model/                 # B：SVM、决策树及后续模型实现
├── train/
│   ├── config.example.json
│   ├── contracts.py       # 数据集、模型的接口约定
│   └── runner.py          # 共用训练/评估与结果落盘框架
├── runs/                  # 自动生成；不提交
├── weights/               # 自动保存训练权重；权重文件不提交
├── README.md
├── requirements.txt
└── 实验接口框架.md
```

`train/runner.py` 接受数据提供器和模型适配器。SVM、决策树和 CNN 均位于 `model/`。当前仍需 A 实现具体 ImageNet 数据提供器，并生成固定划分清单。

## 环境与模型

安装依赖：

```bash
python -m pip install -r requirements.txt
```

基础模型为 LinearSVC 支持向量机和决策树。SVM 默认使用训练集内拟合的标准化步骤；调参时通过配置中的 `params` 传给 scikit-learn，例如 SVM 的 `C`、决策树的 `max_depth`。ImageNet 规模下请先固定可运行的共同子集和特征尺寸；传统模型会把该子集的特征矩阵放入内存，并非流式拟合。

CNN 为从零训练的双层卷积网络，不下载预训练权重。A 提供 `dataset` 和 `splits` 后，在项目入口中调用：

```python
from model import build_model
from train.runner import run_experiment

result = run_experiment(
    dataset=dataset, split_manifest=splits, model=build_model("cnn"),
    params={"epochs": 5, "lr": 0.001}, batch_size=32, seed=42,
    run_id="cnn-baseline-seed42", eval_split="val",
)
```

CNN 通过 `iter_batches` 逐批读取 `[B, 3, H, W]` 的 RGB 图像，不展平、不缓存整个划分。传统模型保存为 `weights/<run_id>.joblib`，CNN 保存为 `weights/<run_id>.pt`。三者都在 `runs/<run_id>/` 输出相同格式的配置、指标和预测表。

`train/config.example.json` 中的字段说明和模型超参数说明见 [train/CONFIG.md](train/CONFIG.md)。

## 数据访问注意

“流式”表示训练程序按需读取样本，不预先把全部图片保存为本地数据集副本；样本仍需通过网络传输，并会占用批次级内存。具体能否直接流式访问取决于数据托管端和授权。若用 Hugging Face 或其他镜像，需确认该数据集的访问条件、版本和来源符合课程用途及其条款。

原始数据、数据缓存、运行结果、权重及本地路径配置默认不纳入版本控制。只提交可复现所需的代码、公开划分清单（不包含受限数据本身）、数据版本/样本 ID 说明及实验配置。
