import torch
import torch.nn as nn
from torchvision import models

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(ResNet50FeatureExtractor, self).__init__()

        # Load the full ResNet50 model
        resnet = models.resnet50(pretrained=pretrained)

        # Remove the classification head (fc layer)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Removes the last fc layer

        # Optional: freeze layers for transfer learning
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Final output will be [batch_size, 2048, 1, 1]

    def forward(self, x):
        x = self.features(x)            # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)       # Flatten to [B, 2048]
        return x
