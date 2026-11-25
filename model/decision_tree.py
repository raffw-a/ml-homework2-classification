"""决策树分类器模型适配器。"""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.tree import DecisionTreeClassifier

from model._classical_base import ClassicalModelAdapter


class DecisionTreeAdapter(ClassicalModelAdapter):
    """构造并训练带固定随机种子的多分类决策树。"""

    name = "decision_tree"

    def _create_estimator(self, params: Mapping[str, Any], seed: int) -> Any:
        """构造 sklearn.tree.DecisionTreeClassifier。

        params 会传给 DecisionTreeClassifier，例如：
        - max_depth：树的最大深度；越小越简单，可能降低过拟合但也可能欠拟合。
        - criterion：划分质量标准，常用 "gini" 或 "entropy"。
        - min_samples_split：内部节点继续分裂所需的最少样本数。
        - min_samples_leaf：叶节点最少样本数；增大通常会让边界更平滑。
        - max_features：每次寻找划分时考虑的特征数；可限制复杂度/加速。
        - class_weight：类别权重，可设 "balanced" 应对类别不平衡。
        random_state 由统一 seed 控制；缺省 max_depth=20 作为复杂度上限。
        """
        options = dict(params)
        # 固定随机状态；用户未指定树深时使用较保守的默认上限。
        options["random_state"] = seed
        options.setdefault("max_depth", 20)
        return DecisionTreeClassifier(**options)
