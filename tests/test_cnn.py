"""用合成图像验证 CNN 与共用训练框架的接口，不访问网络。"""

import csv
import random
import tempfile
import unittest
from pathlib import Path

import torch

from model import build_model
from model.cnn import ImageCNN
from train.contracts import Batch
from train.runner import run_experiment


class Images:
    dataset_id = "synthetic"
    dataset_version = "1"
    class_names = ["red", "green", "blue"]

    def __init__(self):
        self.calls = []

    def labels(self, ids):
        return [int(i) % 3 for i in ids]

    def iter_batches(self, ids, *, batch_size, shuffle, seed):
        self.calls.append((list(ids), shuffle, seed))
        ordered = list(ids)
        if shuffle:
            random.Random(seed).shuffle(ordered)
        for i in range(0, len(ids), batch_size):
            keys = ordered[i:i + batch_size]
            labels = self.labels(keys)
            x = torch.zeros(len(keys), 3, 16, 16, dtype=torch.uint8)
            x[range(len(keys)), labels] = 255
            yield Batch(keys, x, labels)


class CNNTest(unittest.TestCase):
    def test_train_evaluate_and_reload(self):
        data = Images()
        splits = {"train": list("01234"), "val": list("567"), "test": ["8"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_experiment(
                dataset=data, model=build_model("cnn"), split_manifest=splits,
                params={"epochs": 2, "device": "cpu"}, batch_size=2, seed=42,
                run_id="cnn", output_dir=root / "runs", weights_dir=root / "weights",
            )
            self.assertEqual(data.calls, [(splits["train"], True, 42),
                                          (splits["train"], True, 43), (splits["val"], False, 0)])
            self.assertEqual(result["metrics"]["num_samples"], 3)
            with (root / "runs/cnn/predictions.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["sample_id"] for r in rows], splits["val"])
            checkpoint = torch.load(root / "weights/cnn.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["class_names"], data.class_names)
            net = ImageCNN(3).eval()
            net.load_state_dict(checkpoint["state_dict"])
            with torch.no_grad():
                predictions = [p for batch in data.iter_batches(splits["val"], batch_size=2, shuffle=False, seed=0)
                               for p in net(batch.x.float() / 255).argmax(1).tolist()]
            self.assertEqual(predictions, [int(r["y_pred"]) for r in rows])
