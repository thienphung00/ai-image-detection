# config.yaml

project:
  root_dir: ".."  # Relative to the current file; use absolute path if needed

paths:
  dataset:
    train_dir: "data/DF40_train"
    test_dir: "data/DF40_test"
  output:
    checkpoint_dir: "training/detectors"
    log_dir: "logs"
    results_dir: "results"

hyperparameters:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.0001
  img_size: [224, 224]  # H x W

models:
  resnet_name: "resnet50"
  swin_name: "swin"
  fusion_model_name: "resnet_swin_fusion"

logging:
  level: "INFO"
  use_tensorboard: true

runtime:
  device: "cuda"  # Can dynamically switch in Python with torch.cuda.is_available()
