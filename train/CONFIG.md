# 配置字段说明

`config.example.json` 是 JSON 格式，JSON 不支持注释，因此字段解释单独写在这里。

## dataset

- `provider`：未来由 A 实现的数据适配器名称；当前是占位值，不可直接导入运行。
- `dataset_id`：数据集标识，例如 `imagenet-1k`。
- `dataset_version`：数据版本、仓库 revision 或文件清单版本；应记录可复现的具体值。
- `access_mode`：数据访问方式说明，例如远端分片流式读取或共享存储。
- `split_manifest`：train/val/test 样本 ID 清单路径，三份 ID 必须互斥。
- `class_map`：类别名与整数标签之间的固定映射。
- `local_data_root`：仅在数据提供器必须读取本地/共享目录时填写机器相关路径；不要提交个人路径。
- `num_classes`：本次任务中类别总数。若只选 ImageNet 子集，此值应等于子集类别数，而不一定是 1000。

## experiment

- `run_id`：本次运行的唯一名称；同名结果目录已存在时 runner 会报错，避免覆盖结果。
- `model`：模型工厂名称；当前可用 `svm`、`decision_tree`、`cnn`。
- `variant`：实验版本标签，例如 `baseline`、`tuned`。
- `seed`：随机种子；用于模型内部可控的随机过程。
- `eval_split`：调参阶段设为 `val`；最终参数冻结后才设为 `test`。
- `batch_size`：CNN 等按批读取数据时每批样本数。传统模型保留该字段以统一接口，不会按批拟合。
- `params`：传给对应 scikit-learn 模型的超参数。SVM 可调 `C`、`max_iter`、`tol`、`class_weight`；决策树可调 `max_depth`、`criterion`、`min_samples_split`、`min_samples_leaf`、`max_features`、`class_weight`。随机种子由 `seed` 控制，不应放在这里。
- `preprocessing.image_size`：数据提供器使用的输入图像尺寸；若做了缩放/裁剪，应记录实际设置。
- `preprocessing.normalization`：记录归一化或标准化方法。若统计量由数据估计，只能用训练集拟合。
- `preprocessing.flatten_for_classical_models`：传统模型是否将图像展平成 `[N, D]` 特征；SVM 和决策树适配器当前需要矩阵输入。

CNN 实验将 `model` 设为 `cnn`，使用新的 `run_id`。`params` 支持 `epochs`（训练轮数，默认 5）、`lr`（Adam 学习率，默认 0.001）、`weight_decay`（权重衰减，默认 0）和 `device`（默认 `auto`，也可指定 `cpu` 或 `cuda`）。数据提供器需保留 RGB 通道维度；`uint8` 输入由模型除以 255，浮点输入直接使用，由 A 负责统一缩放或标准化。网络输出类别数取自 `dataset.class_names`，标签应为从 0 起的连续编号。

## 输出路径

- `output_dir`：指标、配置和逐样本预测结果的根目录，默认 `runs/`。
- `weights_dir`：权重目录，默认 `weights/`。CNN 保存为 `<run_id>.pt`，传统模型保存为 `<run_id>.joblib`。

runner 结果目录包含：

- `config.json`：数据版本、模型名称、参数、预处理和权重路径。
- `metrics.json`：评估划分、样本数、Accuracy 和 Macro-F1。
- `predictions.csv`：每个评估样本的 `sample_id`、真实标签和预测标签。
