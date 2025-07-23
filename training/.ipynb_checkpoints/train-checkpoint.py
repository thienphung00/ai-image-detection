import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
from datetime import timedelta
from tqdm import tqdm

from config import settings
from models.fusion_model import FusionEnsembleModel
from data.loaders.df40_loader import DF40Dataset  # Create this next!
from utils.metrics import compute_metrics  # Placeholder for accuracy, F1, etc.

def train():
    # === Device ===
    device = torch.device(settings.DEVICE)

    # === Model ===
    model = FusionEnsembleModel(fusion_type="attention")  # swap as needed
    model.to(device)

    # === Dataset ===
    train_dataset = DF40Dataset(root_dir=settings.TRAIN_DIR, split='train')
    val_dataset = DF40Dataset(root_dir=settings.TRAIN_DIR, split='val')

    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    # === Optimizer & Loss ===
    criterion = nn.BCELoss()  # Binary output assumed
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(settings.NUM_EPOCHS):
        epoch_start = time.time()
    
        model.train()
        train_loss = 0.0
    
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        avg_train_loss = train_loss / len(train_loader)
    
        # Evaluation step
        model.eval()
        val_preds, val_labels = [], []
    
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
    
        val_metrics = compute_metrics(val_preds, val_labels)
        val_f1 = val_metrics["f1"]
    
        # Checkpoint handling
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), settings.CHECKPOINT_DIR + "/fusion_best.pth")
            checkpoint_status = "✅ Saved new best model"
        else:
            checkpoint_status = "⏸ No improvement"
    
        elapsed = time.time() - start_time
        remaining = (elapsed / (epoch + 1)) * (settings.NUM_EPOCHS - epoch - 1)
    
        print(f"\n📢 Epoch [{epoch+1}/{settings.NUM_EPOCHS}]")
        print(f"🧪 Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")
        print(f"{checkpoint_status}")
        print(f"🕒 Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {timedelta(seconds=int(remaining))}")
    

if __name__ == "__main__":
    train()
