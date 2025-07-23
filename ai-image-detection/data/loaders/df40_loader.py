import os
from PIL import Image
from torch.utils.data import Dataset
from config import settings
from utils.preprocessing import get_transform

class DF40DualInputDataset(Dataset):
    def __init__(self, root_dir_rgb, root_dir_fft, split='train', generator_filter=None):
        self.root_dir_rgb = root_dir_rgb
        self.root_dir_fft = root_dir_fft
        self.rgb_transform = get_transform(model_name="resnet50")
        self.fft_transform = get_fft_transform(model_name="xception")  # You can tune this for FFT input style

        self.samples = []

        generators = [g for g in os.listdir(root_dir_rgb)
                      if os.path.isdir(os.path.join(root_dir_rgb, g))]

        for gen_name in generators:
            if generator_filter and gen_name != generator_filter:
                continue

            gen_path_rgb = os.path.join(root_dir_rgb, gen_name)
            gen_path_fft = os.path.join(root_dir_fft, gen_name)

            if os.path.isdir(gen_path_rgb) and ('ff' in os.listdir(gen_path_rgb) or 'cdf' in os.listdir(gen_path_rgb)):
                for domain in ['ff', 'cdf']:
                    domain_path_rgb = os.path.join(gen_path_rgb, domain)
                    domain_path_fft = os.path.join(gen_path_fft, domain)
                    if not os.path.exists(domain_path_rgb): continue
                    for fname in os.listdir(domain_path_rgb):
                        fpath_rgb = os.path.join(domain_path_rgb, fname)
                        fpath_fft = os.path.join(domain_path_fft, fname)
                        self.samples.append((fpath_rgb, fpath_fft, 1))  # fake label

            else:
                numbered = [d for d in os.listdir(gen_path_rgb)
                            if d.isdigit() and os.path.isdir(os.path.join(gen_path_rgb, d))]
                for sub in numbered:
                    sub_path_rgb = os.path.join(gen_path_rgb, sub)
                    sub_path_fft = os.path.join(gen_path_fft, sub)
                    for fname in os.listdir(sub_path_rgb):
                        fpath_rgb = os.path.join(sub_path_rgb, fname)
                        fpath_fft = os.path.join(sub_path_fft, fname)
                        self.samples.append((fpath_rgb, fpath_fft, 1))

        if split in ['train', 'val']:
            split_idx = int(0.8 * len(self.samples))
            if split == 'train':
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath_rgb, fpath_fft, label = self.samples[idx]

        rgb_image = Image.open(fpath_rgb).convert("RGB")
        fft_image = Image.open(fpath_fft).convert("RGB")

        rgb_tensor = self.rgb_transform(rgb_image)
        fft_tensor = self.fft_transform(fft_image)

        return {
            "rgb": rgb_tensor,
            "fft": fft_tensor,
            "label": label
        }
