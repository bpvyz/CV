import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

def load_model(model_path):
    model_data = joblib.load(model_path)
    return model_data["model"], model_data["scaler"]

def load_image(file_path):
    return cv2.imread(file_path)

def apply_threshold(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    return img_thresh

def find_contours(img_thresh):
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_features(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / w if w != 0 else 0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    moments = cv2.moments(contour)
    cx = moments["m10"] / moments["m00"] if moments["m00"] != 0 else 0 # Centroid x
    cy = moments["m01"] / moments["m00"] if moments["m00"] != 0 else 0 # Centroid y

    return [area, perimeter, aspect_ratio, cx, cy]

def filter_contours_by_features(contours, model, scaler, low_aspect_ratio=1.7, high_aspect_ratio=2.2, threshold = 0.2):
    valid_contours = []

    for contour in contours:
        # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        center = rect[0]
        x, y = int(center[0]), int(center[1])
        width, height = rect[1]

        aspect_ratio = width / height if height != 0 else 0
        if aspect_ratio < 1:
            aspect_ratio = height / width if width != 0 else 0
        print(f"Aspect ratio: {aspect_ratio:.3f}, width: {width:.3f}, height: {height:.3f}, x: {x}, y: {y}")

        if low_aspect_ratio - threshold < aspect_ratio < low_aspect_ratio or high_aspect_ratio + threshold > aspect_ratio > high_aspect_ratio:
            valid_contours.append(contour)

    return valid_contours

# kontura u region of interest da bi moglo da se radi
def process_contour_to_roi(contour, img_thresh):
    x, y, w, h = cv2.boundingRect(contour)
    roi = img_thresh[y:y+h, x:x+w]
    return roi, (x, y, w, h)

def detect_p(image_path, model_path):
    model, scaler = load_model(model_path)
    img_bgr = load_image(image_path)
    img_thresh = apply_threshold(img_bgr)

    contours = find_contours(img_thresh)
    print(f"Found {len(contours)} contours.")

    valid_contours = filter_contours_by_features(contours, model, scaler)
    print(f"Valid contours after filtering: {len(valid_contours)}")

    detected_count = 0
    img_annotated = img_bgr.copy()
    all_probabilities = []

    for contour in valid_contours:
        roi, bbox = process_contour_to_roi(contour, img_thresh)
        roi_resized = cv2.resize(roi, (28, 28))

        roi_contours = find_contours(roi_resized)
        features_list = []
        for roi_contour in roi_contours:
            features = extract_features(roi_contour)
            features_list.append(features)

        if features_list:
            roi_scaled = scaler.transform(features_list)
            probabilities = model.predict_proba(roi_scaled) # predict probability

            p_probability = probabilities[0, 1]
            all_probabilities.append((bbox, p_probability))

            if (0.7 < p_probability < 0.99) or (0.003 < p_probability < 0.01) or (0.06 < p_probability < 0.08):
                detected_count += 1
                cv2.rectangle(img_annotated, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
                cv2.putText(img_annotated, f'P={p_probability:.2f}', (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    print(f"Detected {detected_count} 'P' letters.")

    img_contours = img_thresh.copy()
    img_contours_color = cv2.cvtColor(img_contours, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(img_contours_color, [box], 0, (0, 0, 255), 1)

        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(img_contours_color, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    img_valid_contours = img_thresh.copy()
    img_valid_contours_color = cv2.cvtColor(img_contours, cv2.COLOR_GRAY2BGR)

    for contour in valid_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(img_valid_contours_color, [box], 0, (0, 0, 255), 1)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(img_valid_contours_color, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    print("All Contour Probabilities:")
    for bbox, p_prob in all_probabilities:
        print(f"Bounding Box: {bbox}, 'P' Probability: {p_prob}")

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_contours_color, cv2.COLOR_BGR2RGB))
    plt.title("All Contours with Coordinates")

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_valid_contours_color, cv2.COLOR_BGR2RGB))
    plt.title("Valid Contours with Coordinates")

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
    plt.title("Image with 'P' Predictions")
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "../MaterijalDodatna/ulaz.png"
    model_path = "models/letter_p_detector.pkl"
    detect_p(image_path, model_path)
