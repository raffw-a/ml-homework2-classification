"""线性支持向量机（LinearSVC）模型适配器。"""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from model._classical_base import ClassicalModelAdapter


class LinearSVMAdapter(ClassicalModelAdapter):
    """用标准化特征训练多分类线性 SVM。"""

    name = "svm"

    def _create_estimator(self, params: Mapping[str, Any], seed: int) -> Any:
        """构造 StandardScaler + LinearSVC 的 Pipeline。

        params 会传给 sklearn.svm.LinearSVC，例如：
        - C：正则化强度的倒数。C 越大越强调拟合训练样本，过拟合风险可能上升。
        - max_iter：优化器最大迭代次数；默认 5000，未收敛时可增加。
        - tol：停止优化的容差；调小通常要求更精细收敛、训练可能更久。
        - class_weight：类别权重；可设 "balanced" 缓解类别不平衡。
        random_state 由统一 seed 控制，不接受 params 覆盖。

        StandardScaler 在 Pipeline.fit 中只拟合训练数据，避免验证/测试泄漏。
        """
        options = dict(params)
        # 所有运行使用 runner 传入的种子；缺省迭代上限设为 5000。
        options["random_state"] = seed
        options.setdefault("max_iter", 5000)
        return make_pipeline(
            StandardScaler(),
            LinearSVC(**options),
        )
