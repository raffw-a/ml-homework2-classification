"""共用实验流程：校验划分、调用模型、计算指标并保存结果。

数据提供器和模型实例由项目入口注入，因此本模块不绑定 ImageNet 路径、
下载方式或某一种模型库。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from train.contracts import DatasetProvider, ModelAdapter


def run_experiment(
    *,
    dataset: DatasetProvider,
    model: ModelAdapter,
    split_manifest: Mapping[str, Sequence[str]],
    params: Mapping[str, Any],
    seed: int,
    batch_size: int,
    run_id: str,
    output_dir: str | Path = "runs",
    weights_dir: str | Path = "weights",
    eval_split: str = "val",
    variant: str = "baseline",
    preprocessing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """用固定训练 ID 拟合模型，在验证集或测试集评估并保存实验文件。

    params: 模型超参数，如 SVM 的 C 或决策树的 max_depth。
    seed: 随机种子，传给模型以便复现实验。
    batch_size: 图像批次数量；矩阵模型接口保留该参数以统一调用签名。
    run_id: 本次运行唯一名称，同时用于结果目录和权重文件名。
    eval_split: 调参阶段用 val；最终配置冻结后才使用 test。
    variant: 用于区分 baseline、tuned 等实验版本。
    """
    # 测试集不能用于调参；评估目标只允许验证集或测试集。
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty directory name")
    if model.input_kind not in {"matrix", "batches"}:
        raise ValueError("model.input_kind must be 'matrix' or 'batches'")

    train_ids = list(split_manifest["train"])
    eval_ids = list(split_manifest[eval_split])
    if not train_ids or not eval_ids:
        raise ValueError("train and evaluation splits must be non-empty")
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    # 三个子集必须各自无重复，且两两互斥，防止数据泄漏。
    split_sets = {
        name: set(split_manifest.get(name, [])) for name in ("train", "val", "test")
    }
    for name in ("train", "val", "test"):
        if len(split_sets[name]) != len(split_manifest.get(name, [])):
            raise ValueError(f"sample IDs must be unique within {name}")
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        raise ValueError("train, val, and test sample IDs must be pairwise disjoint")

    # 模型适配器按自己的输入类型取数据：传统模型取矩阵，CNN 按批读取。
    model.fit(
        dataset,
        train_ids,
        val_ids=list(split_manifest.get("val", [])),
        params=params,
        seed=seed,
        batch_size=batch_size,
    )
    # 训练结束即保存模型；结果目录还会记录该权重路径。
    weights_path = Path(weights_dir) / f"{run_id}.joblib"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(weights_path))
    # 预测结果与真实标签都按 eval_ids 顺序排列，用于后续逐样本分析。
    y_pred = np.asarray(
        model.predict(dataset, eval_ids, batch_size=batch_size), dtype=np.int64
    )
    y_true = np.asarray(dataset.labels(eval_ids), dtype=np.int64)

    if len(y_pred) != len(eval_ids) or len(y_true) != len(eval_ids):
        raise ValueError("labels and predictions must match the evaluation ID count")

    # Macro-F1 对每个类别等权平均，类别不平衡时可补充 Accuracy 观察。
    metrics = {
        "split": eval_split,
        "num_samples": len(eval_ids),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=False)

    # 将数据版本、参数和划分信息写入配置，便于复现和 C/D 后续读取。
    config = {
        "run_id": run_id,
        "model": model.name,
        "variant": variant,
        "seed": seed,
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "eval_split": eval_split,
        "num_train_samples": len(train_ids),
        "num_eval_samples": len(eval_ids),
        "params": dict(params),
        "preprocessing": dict(preprocessing or {}),
        "weights_path": str(weights_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "y_true", "y_pred"])
        writer.writerows(zip(eval_ids, y_true.tolist(), y_pred.tolist()))

    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """以 UTF-8 格式写 JSON，供配置和指标结果复用。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
