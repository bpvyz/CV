import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


def compute_fourier_transform(img):
    ft = np.fft.fft2(img)
    return np.fft.fftshift(ft)


def create_amplitude_reduction_mask(magnitude_spectrum, rows, cols, center_radius=30, percentile=99.9,
                                    min_mask_value=0.05):
    crow, ccol = rows // 2, cols // 2

    # threshold za najosvetljenije tacke na spektrumu
    threshold = np.percentile(magnitude_spectrum, percentile)

    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    distance_from_center = np.sqrt(x ** 2 + y ** 2)

    mask = np.ones((rows, cols), dtype=np.float32)

    condition = (magnitude_spectrum > threshold) & (distance_from_center > center_radius)

    if np.any(condition):
        values_to_modify = magnitude_spectrum[condition]

        # boze sacuvaj
        reduction_factors = 1 - np.clip(
            np.log1p(values_to_modify - threshold) / np.log1p(np.max(values_to_modify) - threshold),
            0, 1 - min_mask_value
        )

        mask[condition] = reduction_factors

    return mask

def apply_filter(ft_shifted, mask):
    filtered_ft = ft_shifted * mask
    ft_ishifted = np.fft.ifftshift(filtered_ft)
    img_filtered = np.fft.ifft2(ft_ishifted)
    return np.abs(img_filtered), filtered_ft


def plot_results(original_img, magnitude_spectrum, mask, filtered_img, filtered_ft):
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original img')

    plt.subplot(122)
    plt.imshow(np.log(magnitude_spectrum + 1), cmap="magma")
    plt.title('Magnitude Spectrum')

    plt.show()

    plt.imshow(np.log(mask), cmap="binary")
    plt.show()

    plt.subplot(121)
    plt.imshow(filtered_img, cmap='gray')
    plt.title('Filtered img')

    plt.subplot(122)
    plt.imshow(np.log(np.abs(filtered_ft) + 1), cmap="magma")
    plt.title('Filtered Magnitude Spectrum')

    plt.show()


def main():
    img = load_image(f"MaterijalLV1/slika_{18823 % 5}.png")
    rows, cols = img.shape

    ft_shifted = compute_fourier_transform(img)
    magnitude_spectrum = np.abs(ft_shifted)

    mask = create_amplitude_reduction_mask(magnitude_spectrum, rows, cols)
    img_filtered, filtered_ft = apply_filter(ft_shifted, mask)

    plot_results(img, magnitude_spectrum, mask, img_filtered, filtered_ft)


if __name__ == "__main__":
    main()