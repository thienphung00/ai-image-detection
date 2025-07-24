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

def get_fft_transform(model_name="xception"):
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.CenterCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # center-normalized
                             std=[0.5, 0.5, 0.5])   # less biased by ImageNet stats
    ])



def get_fft_transform(model_name="vit-base-patch16-224"):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # center-balanced for FFT magnitude
                             std=[0.5, 0.5, 0.5])   # neutral to ImageNet biases
    ])
