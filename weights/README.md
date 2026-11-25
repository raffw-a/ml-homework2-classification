# Trained weights

The experiment runner writes fitted model artifacts here, named
`<run_id>.joblib`. Generated artifacts are ignored by Git; retain the matching
`runs/<run_id>/config.json` and split manifest to reproduce an experiment.

Do not commit ImageNet images, credentials, or large checkpoints to this folder.
