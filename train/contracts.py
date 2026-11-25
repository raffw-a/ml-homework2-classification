"""Shared interfaces for dataset providers and model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Batch:
    """CNN 用的图像批次：x 为 [B, C, H, W]，y 为类别整数标签。"""

    sample_ids: Sequence[str]  # 稳定样本 ID，供预测结果和清单对齐
    x: Any  # 图像张量或数组
    y: Any  # [B] 标签


@dataclass(frozen=True)
class MatrixSplit:
    """SVM、决策树等 scikit-learn 模型使用的内存特征矩阵。"""

    sample_ids: Sequence[str]  # 必须与 x、y 的行顺序一致
    x: Any  # [N, D]，N 个样本、D 个特征
    y: Any  # [N]，每个样本对应一个整数类别


class DatasetProvider(Protocol):
    """数据适配器接口；具体数据源、授权和机器路径由 A 侧实现。"""

    dataset_id: str  # 数据集名称或稳定标识
    dataset_version: str  # 数据版本/修订号，用于复现实验
    class_names: Sequence[str]  # 整数标签到类别名称的固定映射

    def load_split_manifest(self, path: str) -> Mapping[str, Sequence[str]]:
        """读取固定的 train/val/test 样本 ID；模型不得自行重新划分。"""
        ...

    def labels(self, sample_ids: Sequence[str]) -> Any:
        """按传入 ID 的原顺序返回整数标签。"""
        ...

    def as_matrix(self, sample_ids: Sequence[str]) -> MatrixSplit:
        """返回传统模型需要的 [N, D] 特征；大数据集应只传入固定子集。"""
        ...

    def iter_batches(
        self,
        sample_ids: Sequence[str],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterable[Batch]:
        """按批读取图像；验证/测试不能打乱顺序或丢弃最后一个不满批次。"""
        ...


class ModelAdapter(Protocol):
    """模型适配器接口；每种算法在 model/ 中独立实现。"""

    name: str
    input_kind: str  # "matrix" 表示传统模型；"batches" 表示 CNN

    def fit(
        self,
        dataset: DatasetProvider,
        train_ids: Sequence[str],
        *,
        val_ids: Sequence[str],
        params: Mapping[str, Any],
        seed: int,
        batch_size: int,
    ) -> None:
        """拟合模型；只能用 train_ids 更新参数，val_ids 仅可用于监控/早停。"""
        ...

    def predict(
        self,
        dataset: DatasetProvider,
        sample_ids: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[int]:
        """按 sample_ids 原顺序返回预测类别，每个 ID 必须有且仅有一个结果。"""
        ...

    def save(self, path: str) -> None:
        """保存已拟合模型到 runner 指定的权重路径。"""
        ...
