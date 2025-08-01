import torch
import torch.nn as nn
from models.resnet50 import ResNet50FeatureExtractor
from models.xception import XceptionFeatureExtractor
from models.fusion_blocks import ConcatFusion, WeightedSumFusion, AttentionFusion

class FusionEnsembleModel(nn.Module):
    def __init__(self, fusion_type="concat", num_classes=1):
        super(FusionEnsembleModel, self).__init__()

        self.resnet = ResNet50FeatureExtractor(pretrained=True, freeze=True)
        self.xception = XceptionFeatureExtractor(pretrained=True, freeze=True)

        # Choose fusion strategy
        if fusion_type == "concat":
            self.fusion = ConcatFusion()
            fusion_dim = 4096
        elif fusion_type == "weighted":
            self.fusion = WeightedSumFusion(feature_dim=2048)
            fusion_dim = 2048
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(feature_dim=2048)
            fusion_dim = 2048
        else:
            raise ValueError(f"Fusion type '{fusion_type}' not recognized.")

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat_r = self.resnet(x)
        feat_x = self.xception(x)
        fused = self.fusion(feat_r, feat_x)
        return self.classifier(fused)
