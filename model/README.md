# Model implementations

The `svm.py` and `decision_tree.py` adapters are implemented. Add subsequent
model files here, for example `cnn.py`. Do not put dataset paths or split logic
in model files.

Each adapter should satisfy `ModelAdapter` in `train/contracts.py`:

- `name`: stable model name for result files.
- `input_kind`: `"matrix"` for classical estimators or `"batches"` for CNN.
- `fit(dataset, train_ids, ..., params, seed, batch_size)`.
- `predict(dataset, sample_ids, ..., batch_size)`: one prediction per ID, in ID order.

All preprocessing learned from data must be fit on training data only. SVM and
decision-tree adapters use `dataset.as_matrix(ids)`; CNN should use
`dataset.iter_batches(...)`. For ImageNet, classical models should use the same
predeclared, manageable sample subset as other models; batching alone does not
make a scikit-learn estimator out-of-core.
