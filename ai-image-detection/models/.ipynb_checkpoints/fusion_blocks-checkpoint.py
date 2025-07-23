import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatFusion(nn.Module):
    """
    Simple feature concatenation: [ResNet feat | Xception feat]
    Output: [batch_size, 4096]
    """
    def __init__(self):
        super(ConcatFusion, self).__init__()

    def forward(self, feat1, feat2):
        return torch.cat((feat1, feat2), dim=1)


class WeightedSumFusion(nn.Module):
    """
    Weighted sum fusion with learnable alpha.
    Assumes both features have same size (e.g. 2048).
    Output: [batch_size, 2048]
    """
    def __init__(self, feature_dim=2048):
        super(WeightedSumFusion, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)  # initialized at 0.5

    def forward(self, feat1, feat2):
        return self.alpha * feat1 + (1 - self.alpha) * feat2


class AttentionFusion(nn.Module):
    """
    Attention-based fusion: learns relevance weights for each feature vector
    Output: [batch_size, feature_dim]
    """
    def __init__(self, feature_dim=2048):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, feat1, feat2):
        fused = torch.cat((feat1, feat2), dim=1)
        weights = self.attention(fused)  # shape: [batch_size, 2]
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        return w1 * feat1 + w2 * feat2
