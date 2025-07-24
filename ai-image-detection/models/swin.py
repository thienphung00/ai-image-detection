import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class SwinFeatureExtractor(nn.Module):
    def __init__(self, model_name='microsoft/swinv2-tiny-patch4-window8-256', freeze=True):
        super(SwinFeatureExtractor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.swin = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.swin.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: FFT-transformed input tensor of shape [B, 3, H, W]
        Returns: CLS token embeddings of shape [B, hidden_size]
        """
        # Convert tensor to PIL images for processor
        images = [img for img in x]  # assumes x is a batch of tensors

        # Preprocess using HuggingFace processor
        inputs = self.processor(images=images, return_tensors="pt").to(self.swin.device)

        # Forward pass
        outputs = self.swin(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        return cls_embeddings
