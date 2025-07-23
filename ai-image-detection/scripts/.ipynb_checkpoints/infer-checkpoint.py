import os
import torch
from PIL import Image
from torchvision import transforms
import argparse

from models.fusion_model import FusionEnsembleModel
from config import settings

# === Inference Preprocessing ===
transform = transforms.Compose([
    transforms.Resize(settings.IMG_SIZE),
    transforms.CenterCrop(settings.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def infer_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prob = output.item()
        label = "FAKE" if prob > 0.5 else "REAL"
        return label, prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, default=os.path.join(settings.CHECKPOINT_DIR, 'fusion_best.pth'), help="Path to model weights")
    args = parser.parse_args()

    device = torch.device(settings.DEVICE)
    model = FusionEnsembleModel(fusion_type="attention")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    label, confidence = infer_image(args.image, model, device)
    print(f"\n🔍 Prediction: {label} ({confidence:.4f} confidence)")

if __name__ == "__main__":
    main()
