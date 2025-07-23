import torch
import torch.nn as nn
import timm  # HuggingFace's PyTorch Image Models

class XceptionFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(XceptionFeatureExtractor, self).__init__()

        # Load Xception model from timm
        self.xception = timm.create_model('xception', pretrained=pretrained, num_classes=0)  # Removes FC head

        # Optional: Freeze parameters
        if freeze:
            for param in self.xception.parameters():
                param.requires_grad = False

        # Final output will be [batch_size, 2048]

    def forward(self, x):
        x = self.xception(x)  # Already flattened by timm model
        return x
