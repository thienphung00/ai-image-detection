import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224', freeze=True):
        super(ViTFeatureExtractor, self).__init__()
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.vit = ViTModel.from_pretrained(model_name)

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: FFT-transformed input tensor of shape [B, 3, H, W] (e.g. 224x224 RGB image from FFT)
        Returns: CLS token embeddings of shape [B, hidden_size]
        """

        # Convert tensor to pixel values expected by ViT processor
        inputs = self.processor(images=[xi for xi in x], return_tensors="pt")

        # Move inputs to same device as model
        inputs = {k: v.to(self.vit.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.vit(**inputs)

        # Extract CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return cls_embeddings
