import os
import numpy as np
from PIL import Image
import cv2

def compute_fft_magnitude(image: Image.Image) -> Image.Image:
    # Convert to grayscale for frequency analysis
    gray = np.array(image.convert("L"))

    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)

    # Normalize to 0–255 and convert to 3 channels
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    fft_rgb = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return Image.fromarray(fft_rgb)

def process_folder(input_dir: str, output_dir: str):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                # Ensure output folder exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    image = Image.open(input_path)
                    fft_image = compute_fft_magnitude(image)
                    fft_image.save(output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply FFT and save frequency-domain images")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to original RGB images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save FFT-transformed images")

    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)

