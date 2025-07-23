import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_metrics(preds, labels):
    """
    preds: list or numpy array of predicted probabilities (floats between 0 and 1)
    labels: list or numpy array of ground truth labels (0 or 1)
    """
    preds = np.array(preds)
    labels = np.array(labels).astype(int)

    # Threshold to get binary predictions
    binary_preds = (preds > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, preds)
    }

    return metrics
