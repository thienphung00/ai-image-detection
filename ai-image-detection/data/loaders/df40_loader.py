class DF40Dataset(Dataset):
    def __init__(self, root_dir, split='train', generator_filter=None, model_name="resnet50"):
        self.root_dir = root_dir
        self.transform = (
            get_fft_transform(model_name) 
            if "swin" in model_name.lower() 
            else get_transform(model_name)
        )
        self.samples = []

        # Load generator folders
        generators = [g for g in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, g))]

        for gen_name in generators:
            if generator_filter and gen_name != generator_filter:
                continue

            gen_path = os.path.join(root_dir, gen_name)

            # Case 1: structured subfolders (e.g. ff/cdf)
            domains = ['ff', 'cdf']
            if any(d in os.listdir(gen_path) for d in domains):
                for domain in domains:
                    domain_path = os.path.join(gen_path, domain)
                    if not os.path.exists(domain_path): continue
                    for fname in os.listdir(domain_path):
                        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                        fpath = os.path.join(domain_path, fname)
                        label = 0 if gen_name.lower() == "original" else 1
                        self.samples.append((fpath, label))
            else:
                # Case 2: numbered folders (e.g. 001–999)
                numbered = [d for d in os.listdir(gen_path) 
                            if d.isdigit() and os.path.isdir(os.path.join(gen_path, d))]
                for sub in numbered:
                    sub_path = os.path.join(gen_path, sub)
                    for fname in os.listdir(sub_path):
                        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                        fpath = os.path.join(sub_path, fname)
                        label = 1
                        self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        image = Image.open(fpath).convert('RGB')
        return self.transform(image), torch.tensor(label, dtype=torch.float)
