import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import settings
from utils.preprocessing import get_transform


class DF40Dataset(Dataset):
    def __init__(self, root_dir, split='train', generator_filter=None, transform=None):
        """
        root_dir: path to dataset folder (e.g. DF40_train or DF40_test)
        split: 'train', 'val', or 'test'
        generator_filter: if provided, only loads data from specific generator (e.g. 'StyleGAN3')
        transform: torchvision transforms
        """
        self.root_dir = root_dir
        self.transform = transform or get_transform(model_name="resnet50")

        self.samples = []

        generators = [g for g in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, g))]
        for gen_name in generators:
            if generator_filter and gen_name != generator_filter:
                continue

            gen_path = os.path.join(root_dir, gen_name)

            # Case 1: structured domains (e.g. ff/cdf)
            if os.path.isdir(gen_path) and ('ff' in os.listdir(gen_path) or 'cdf' in os.listdir(gen_path)):
                for domain in ['ff', 'cdf']:
                    domain_path = os.path.join(gen_path, domain)
                    if not os.path.exists(domain_path): continue
                    for fname in os.listdir(domain_path):
                        fpath = os.path.join(domain_path, fname)
                        self.samples.append((fpath, 1))  # fake label

            # Case 2: numbered folders (e.g. 001–999)
            else:
                numbered = [d for d in os.listdir(gen_path) 
                            if d.isdigit() and os.path.isdir(os.path.join(gen_path, d))]
                for sub in numbered:
                    sub_path = os.path.join(gen_path, sub)
                    for fname in os.listdir(sub_path):
                        fpath = os.path.join(sub_path, fname)
                        self.samples.append((fpath, 1))  # fake label

        # Split dataset if needed
        if split in ['train', 'val']:
            split_idx = int(0.8 * len(self.samples))
            if split == 'train':
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        image = Image.open(fpath).convert("RGB")
        return self.transform(image), label
