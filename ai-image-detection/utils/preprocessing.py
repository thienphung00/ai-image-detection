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

def get_fft_transform(model_name):
    if "xception" in model_name.lower():
        size = 299
    elif "swin" in model_name.lower():
        size = 256
    else:  # defaults to ViT
        size = 224

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # center-balanced for FFT magnitude
                             std=[0.5, 0.5, 0.5])   # avoids ImageNet or dataset bias
    ])
