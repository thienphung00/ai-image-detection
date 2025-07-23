import os
import torch

# ==== Project Root ====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ==== Dataset Paths ====
TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'DF40_train')
TEST_DIR  = os.path.join(ROOT_DIR, 'data', 'DF40_test')

# ==== Output Paths ====
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'training', 'detectors')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# ==== Model Hyperparameters ====
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = (224, 224)

# ==== Model Settings ====
RESNET_NAME = 'resnet50'
XCEPTION_NAME = 'xception'
FUSION_MODEL_NAME = 'resnet_xception_fusion'

# ==== Logging ====
LOGGING_LEVEL = 'INFO'
USE_TENSORBOARD = True

# ==== Device ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
