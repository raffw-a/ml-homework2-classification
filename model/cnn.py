"""按固定样本清单分批训练的 PyTorch 图像分类器，不使用预训练权重。"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from train.contracts import Batch, DatasetProvider


class ImageCNN(nn.Sequential):
    """两层卷积提取局部特征，全局平均池化后输出各类别的未归一化分数。"""

    def __init__(self, num_classes: int):
        super().__init__(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes),
        )


@dataclass
class CNNParams:
    """epochs 是完整训练轮数；lr 是 Adam 学习率；weight_decay 是权重衰减系数。"""

    epochs: int = 5
    lr: float = 0.001
    weight_decay: float = 0.0
    device: str = "auto"  # auto 优先选择 CUDA，否则使用 CPU


def _images(batch: Batch, device: torch.device) -> torch.Tensor:
    """读取 RGB 批次；字节像素缩放到 [0, 1]，浮点输入视为已预处理。"""
    x = torch.as_tensor(batch.x, device=device)
    if x.ndim != 4 or x.shape[:2] != (len(batch.sample_ids), 3):
        raise ValueError("CNN expects RGB images with shape [B, 3, H, W]")
    if x.dtype == torch.uint8:
        return x.float().div_(255)
    if not x.is_floating_point():
        raise ValueError("CNN expects uint8 or floating-point images")
    return x.float()


class CNNAdapter:
    name = "cnn"
    input_kind = "batches"
    weights_suffix = ".pt"

    def __init__(self):
        self.network: ImageCNN | None = None
        self.class_names: list[str] = []
        self.device = torch.device("cpu")

    def fit(self, dataset: DatasetProvider, train_ids: Sequence[str], *,
            val_ids: Sequence[str], params: Mapping[str, Any], seed: int,
            batch_size: int) -> None:
        """每轮重新读取并打乱训练批次；不使用验证集或测试集更新权重。"""
        cfg = CNNParams(**params)
        if not isinstance(cfg.epochs, int) or cfg.epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive integers")
        if not train_ids or len(dataset.class_names) < 2:
            raise ValueError("CNN requires training IDs and at least two classes")
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        )
        torch.manual_seed(seed)
        self.class_names = list(dataset.class_names)
        network = ImageCNN(len(self.class_names)).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(cfg.epochs):
            remaining = set(train_ids)
            for batch in dataset.iter_batches(train_ids, batch_size=batch_size, shuffle=True, seed=seed + epoch):
                # 允许打乱，但每轮必须恰好使用清单指定的样本，包括最后的小批次。
                ids = set(batch.sample_ids)
                if not ids or len(ids) != len(batch.sample_ids) or not ids <= remaining:
                    raise ValueError("unexpected or duplicate training IDs")
                remaining.difference_update(ids)
                x = _images(batch, self.device)
                y = torch.as_tensor(batch.y, device=self.device)
                if y.shape != (len(x),) or y.is_floating_point() or y.is_complex() or y.dtype == torch.bool:
                    raise ValueError("CNN expects one integer class label per image")
                if y.min().item() < 0 or y.max().item() >= len(self.class_names):
                    raise ValueError("class label outside dataset.class_names")
                optimizer.zero_grad()
                loss_fn(network(x), y.long()).backward()
                optimizer.step()
            if remaining:
                raise ValueError("training batches omitted requested sample IDs")
        self.network = network.eval()

    @torch.inference_mode()
    def predict(self, dataset: DatasetProvider, sample_ids: Sequence[str], *, batch_size: int) -> list[int]:
        """按清单顺序推理，不计算梯度，确保输出可以和真实标签逐行比较。"""
        if self.network is None:
            raise RuntimeError("fit the CNN before predicting")
        predictions: list[int] = []
        for batch in dataset.iter_batches(sample_ids, batch_size=batch_size, shuffle=False, seed=0):
            start = len(predictions)
            if list(batch.sample_ids) != list(sample_ids[start:start + len(batch.sample_ids)]):
                raise ValueError("evaluation batches must preserve the requested ID order")
            scores = self.network(_images(batch, self.device))
            predictions.extend(scores.argmax(dim=1).tolist())
        if len(predictions) != len(sample_ids):
            raise ValueError("evaluation batches omitted requested sample IDs")
        return predictions

    def save(self, path: str) -> None:
        """保存权重和有序类别名，便于重建分类头并解释预测编号。"""
        if self.network is None:
            raise RuntimeError("fit the CNN before saving")
        torch.save({"state_dict": self.network.state_dict(), "class_names": self.class_names}, path)
