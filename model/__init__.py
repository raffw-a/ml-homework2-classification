"""模型工厂：把配置中的短名称映射为对应的模型适配器。"""

from model.decision_tree import DecisionTreeAdapter
from model.svm import LinearSVMAdapter


def build_model(name: str):
    """按配置名称创建全新的未训练模型，避免不同实验意外共享状态。"""
    if name == "cnn":
        from model.cnn import CNNAdapter

        return CNNAdapter()
    factories = {
        "svm": LinearSVMAdapter,
        "decision_tree": DecisionTreeAdapter,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted([*factories, 'cnn'])}"
        ) from exc


__all__ = ["DecisionTreeAdapter", "LinearSVMAdapter", "build_model"]
