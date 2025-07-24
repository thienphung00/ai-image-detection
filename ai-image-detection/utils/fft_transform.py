import numpy as np
import cv2
from PIL import Image

def fft_magnitude_transform(image_stream):
    img = Image.open(image_stream).convert('L')
    img_np = np.array(img)

    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
    fft_rgb = cv2.merge([magnitude_spectrum] * 3)

    return Image.fromarray(fft_rgb)
