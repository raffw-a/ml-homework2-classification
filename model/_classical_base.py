"""传统模型适配器共用的数据校验、拟合、预测和权重保存逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from train.contracts import DatasetProvider, MatrixSplit


class ClassicalModelAdapter:
    # scikit-learn 模型要求一次接收完整的二维特征矩阵。
    input_kind = "matrix"

    def __init__(self) -> None:
        """创建尚未训练的适配器；具体估计器在 fit 时生成。"""
        self.estimator: Any | None = None

    def _matrix(
        self, dataset: DatasetProvider, sample_ids: Sequence[str]
    ) -> MatrixSplit:
        """通过数据提供器读取特征，并验证形状、ID 顺序及数值有效性。"""
        data = dataset.as_matrix(sample_ids)
        if list(data.sample_ids) != list(sample_ids):
            raise ValueError("as_matrix must preserve the requested sample ID order")
        x = np.asarray(data.x)
        y = np.asarray(data.y).reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"classical model input must be [N, D], got {x.shape}")
        if len(x) != len(sample_ids) or len(y) != len(sample_ids):
            raise ValueError("matrix features and labels must match the requested IDs")
        if not np.isfinite(x).all():
            raise ValueError("feature matrix contains NaN or infinite values")
        return MatrixSplit(sample_ids=list(sample_ids), x=x, y=y)

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
        """只用 train_ids 取矩阵并训练；验证集不参与模型拟合。"""
        del val_ids, batch_size
        data = self._matrix(dataset, train_ids)
        self.estimator = self._create_estimator(params, seed)
        self.estimator.fit(data.x, data.y)

    def predict(
        self,
        dataset: DatasetProvider,
        sample_ids: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[int]:
        """读取给定 ID 的矩阵并返回对应预测，顺序保持一致。"""
        del batch_size
        if self.estimator is None:
            raise RuntimeError("model must be fitted before predict")
        data = self._matrix(dataset, sample_ids)
        return self.estimator.predict(data.x).tolist()

    def save(self, path: str) -> None:
        """使用 joblib 序列化 sklearn Pipeline/估计器。"""
        if self.estimator is None:
            raise RuntimeError("model must be fitted before save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, target)

    def _create_estimator(self, params: Mapping[str, Any], seed: int) -> Any:
        """由具体模型子类实现，返回尚未拟合的 sklearn 估计器。"""
        raise NotImplementedError
