import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from config import settings
from models.fusion_model import FusionEnsembleModel
from data.loaders.df40_loader import DF40Dataset  # Assumed data loader

def evaluate(model_path, split='test'):
    device = torch.device(settings.DEVICE)

    # === Load Model ===
    model = FusionEnsembleModel(fusion_type="attention")  # Match what you trained with
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # === Load Dataset ===
    dataset = DF40Dataset(root_dir=settings.TEST_DIR if split == 'test' else settings.TRAIN_DIR, split=split)
    dataloader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            preds = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # === Metrics ===
    all_preds_bin = (np.array(all_preds) > 0.5).astype(int)
    all_labels = np.array(all_labels).astype(int)

    acc = accuracy_score(all_labels, all_preds_bin)
    prec = precision_score(all_labels, all_preds_bin)
    rec = recall_score(all_labels, all_preds_bin)
    f1 = f1_score(all_labels, all_preds_bin)
    auc = roc_auc_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds_bin)

    print("\n🔍 Evaluation Results")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    model_path = os.path.join(settings.CHECKPOINT_DIR, "fusion_best.pth")
    evaluate(model_path)
