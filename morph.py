import matplotlib.pyplot as plt
import cv2

def load_image(file_path):
    return cv2.imread(file_path)

def plot_images(images, titles):
    n = len(images)
    rows = (n + 2) // 3
    plt.figure(figsize=(20, 4 * rows))
    for i in range(n):
        plt.subplot(rows, 3, i + 1)
        # Postavi colormap na 'magma' za poslednje dve slike, inače na 'gray'
        if i >= n - 2:
            cmap = 'magma'
        else:
            # Postavi colormap na 'gray' za 2D slike ili None za RGB slike
            if len(images[i].shape) == 2:
                cmap = 'gray'
            else:
                cmap = None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()

def apply_gray_filters(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_gray_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_gray_closed = cv2.morphologyEx(img_gray_thresh, cv2.MORPH_CLOSE, kernel_close)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_gray_dilated = cv2.dilate(img_gray_closed, kernel_dilate)

    return blur, img_gray_thresh, img_gray_closed, img_gray_dilated

def apply_saturation_filters(saturation_channel):
    _, img_saturation_thresh = cv2.threshold(saturation_channel, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_saturation_median_filtered = cv2.medianBlur(img_saturation_thresh, 7)

    kernel_saturation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_saturation_dilated = cv2.dilate(img_saturation_median_filtered, kernel_saturation)

    return img_saturation_thresh, img_saturation_median_filtered, img_saturation_dilated

def main():
    img_bgr = load_image("MaterijalLV2/coins.png")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # apply gray filters
    blur, img_gray_thresh, img_gray_closed, img_gray_dilated = apply_gray_filters(img_gray)

    # gray diffs
    img_difference = cv2.absdiff(img_gray_thresh, img_gray_closed)
    img_difference2 = cv2.absdiff(img_gray_thresh, img_gray_dilated)

    images = [cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), img_gray, blur, img_gray_thresh, img_gray_closed,
              img_gray_dilated, img_difference, img_difference2]
    titles = [
        "1: Originalna slika (RGB)",
        "2: Grayscale slika",
        "3: Gaussian Blurred (Veličina jezgra: 5x5)",
        "4: Zagušena (Otsu metoda)",
        "5: Morfološko zatvaranje (Veličina jezgra: 7x7)",
        "6: Morfološka dilatacija (Veličina jezgra: 3x3)",
        "7: Razlika: Zatvorena - Zagušena",
        "8: Razlika: Dilatirana - Zagušena"
    ]

    plot_images(images, titles)

    # saturation channel
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation_channel = img_hsv[:, :, 1]

    # apply saturation filters
    img_saturation_thresh, img_saturation_median_filtered, img_saturation_dilated = apply_saturation_filters(
        saturation_channel)

    # saturation diffs
    img_difference = cv2.absdiff(img_saturation_thresh, img_saturation_median_filtered)
    img_difference2 = cv2.absdiff(img_saturation_thresh, img_saturation_dilated)

    images_hsv = [img_hsv, saturation_channel, img_saturation_thresh, img_saturation_median_filtered,
                  img_saturation_dilated, img_difference, img_difference2]
    titles_hsv = [
        "1: HSV slika",
        "2: Saturation kanal",
        "3: Saturation threshold (Otsu metoda)",
        "4: Saturation median filter (Veličina kernela: 7x7)",
        "5: Saturation dilatacija (Veličina kernela: 2x2)",
        "6: Razlika: Saturation median filter - Saturation threshold",
        "7: Razlika: Saturation dilatacija - Saturation threshold"
    ]

    plot_images(images_hsv, titles_hsv)

if __name__ == "__main__":
    main()
