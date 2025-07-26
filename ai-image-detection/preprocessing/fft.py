import os, cv2, json
import numpy as np
from pathlib import Path
from error_handlers.validate_fft import is_valid_fft

def fft_transform(frame_path):
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    return magnitude

def process_fft_from_json(json_path, save_root):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for dataset_name in data:
        for category in data[dataset_name]:
            for split in data[dataset_name][category]:
                for compression in data[dataset_name][category][split]:
                    for video, info in data[dataset_name][category][split][compression].items():
                        fft_folder = Path(save_root) / dataset_name / category / split / compression / video
                        fft_folder.mkdir(parents=True, exist_ok=True)
                        for frame_path in info['frames']:
                            fft_data = fft_transform(frame_path)
                            if not is_valid_fft(fft_data):
                                continue
                            filename = Path(frame_path).stem + '.npy'
                            np.save(fft_folder / filename, fft_data)
