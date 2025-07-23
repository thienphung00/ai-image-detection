# utils/preprocessing.py
from torchvision import transforms

def get_transform(model_name="resnet50"):
    size = (224, 224) if model_name == "resnet50" else (299, 299)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
