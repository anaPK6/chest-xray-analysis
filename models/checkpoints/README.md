# checkpoints/

Drop your Kaggle-trained model here (the handoff contract):

- `state_dict.pth`     — trained DenseNet121 weights
- `labels.json`        — label order (list[str])
- `model_config.json`  — {arch, input_size, num_classes, norm, sigmoid}

Then run the app with `MODEL_SOURCE=local`. Produced by
`notebooks/chexpert_train_kaggle.ipynb` (downloaded from `/kaggle/working`).
