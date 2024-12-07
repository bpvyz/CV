# Import the necessary packages
import cv2
import imutils
import time
import matplotlib.pyplot as plt
from GoogLeNet.deep_learning_with_opencv import classify_image

def load_image_rgb(file_path):
    """Load an image in RGB format."""
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def pyramid(image, scale=2, minSize=(180, 180)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)

        # Check if resizing will still meet the minimum size
        if w < minSize[0] or h < minSize[1]:
            break  # Exit the loop if the image is too small

        # Resize the image using cv2
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        yield image

def sliding_window(image, stepSize, windowSize):
    """Slide a window across the image."""
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def main():
    # Load the original image
    image_path = "../MaterijalLV4/download.png"
    image = cv2.imread(image_path)
    crop = image[5:725, 90:1530]  # Crop the image as needed

    # Parameters
    (winW, winH) = (180, 180)
    prototxt = "GoogLeNet/bvlc_googlenet.prototxt"
    model = "GoogLeNet/bvlc_googlenet.caffemodel"
    labels = "GoogLeNet/synset_words.txt"

    # Loop over the image pyramid and sliding windows
    for resized in pyramid(crop, scale=2):
        scale_factor = crop.shape[1] / float(resized.shape[1])  # Scale factor for coordinates
        for (x, y, window) in sliding_window(resized, stepSize=180, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # Classify the window
            label = classify_image(
                window,
                prototxt,
                model,
                labels
            )

            # Determine the label and color
            if "dog" in label:
                color = (0, 255, 255)  # Yellow
                text = "DOG"
            elif "cat" in label:
                color = (0, 0, 255)  # Red
                text = "CAT"
            else:
                continue  # Skip non-dog and non-cat regions

            # Scale the coordinates back to the original crop size
            original_x = int(x * scale_factor)
            original_y = int(y * scale_factor)
            original_w = int(winW * scale_factor)
            original_h = int(winH * scale_factor)

            # Draw the rectangle and label
            cv2.rectangle(crop, (original_x + 2, original_y + 2),
                          (original_x + original_w - 2, original_y + original_h - 2), color, 2)
            cv2.putText(crop, text, (original_x + 5, original_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the final image
    cv2.imshow("Detected Regions", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
