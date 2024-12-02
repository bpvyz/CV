import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import cv2

np.random.seed(42)

# dataset skinuti sa https://www.kaggle.com/datasets/crawford/emnist kao ZIP i unzipovati ga u /datasets/
# zatim pokrenuti model.py pa po zavrsetku dodatni.py

dataset_path = "datasets/archive/emnist-letters-train.csv" # set path to dataset path
model_output_path = "models/letter_p_detector.pkl"  # set path to model path

def extract_features_from_image(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w != 0 else 0
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        moments = cv2.moments(contour)
        cx = moments["m10"] / moments["m00"] if moments["m00"] != 0 else 0
        cy = moments["m01"] / moments["m00"] if moments["m00"] != 0 else 0
        features.append([area, perimeter, aspect_ratio, cx, cy])
    return features

# Load and preprocess the dataset
def load_data(dataset_path):
    print("Loading dataset...")
    data = pd.read_csv(dataset_path, header=None)
    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values
    print(f"Dataset loaded with {images.shape[0]} samples.")
    return images, labels

# Preprocess data
def preprocess_data(images, labels):
    print("Preprocessing data...")
    images = images / 255.0
    labels_binary = np.where(labels == 16, 1, 0)  # 1 for 'P', 0 for non-'P'

    p_indices = np.where(labels_binary == 1)[0]
    non_p_indices = np.where(labels_binary == 0)[0]
    sampled_non_p_indices = np.random.choice(non_p_indices, size=len(p_indices), replace=False)

    balanced_indices = np.concatenate([p_indices, sampled_non_p_indices])
    balanced_images = images[balanced_indices]
    balanced_labels = labels_binary[balanced_indices]

    print(f"Filtered {len(balanced_images)} samples (balanced 'P' vs. 'non-P').")
    return balanced_images, balanced_labels

# Split data
def split_data(images, labels):
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

# Train model
def train_model(X_train, y_train):
    print("Training model...")
    features_list = []
    labels_list = []
    for image, label in zip(X_train, y_train):
        img = image.reshape(28, 28)
        img = (img * 255).astype(np.uint8)

        features = extract_features_from_image(img)

        if features:
            for feature in features:
                features_list.append(feature)
                labels_list.append(label)

    if len(features_list) != len(labels_list):
        print(f"Error: Number of features {len(features_list)} does not match number of labels {len(labels_list)}")
        return None, None

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_list)

    model = SVC(kernel='linear', random_state=42, probability=True)
    model.fit(scaled_features, labels_list)
    print("Model training completed.")
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    print("Evaluating model...")
    features_list = []
    test_labels_list = []

    for image, label in zip(X_test, y_test):
        img = image.reshape(28, 28)
        img = (img * 255).astype(np.uint8)
        features = extract_features_from_image(img)

        if features:
            for feature in features:
                features_list.append(feature)
                test_labels_list.append(label)

    scaled_features = scaler.transform(features_list)
    y_pred = model.predict(scaled_features)

    print("Classification Report:\n", classification_report(test_labels_list, y_pred))
    print("Accuracy:", accuracy_score(test_labels_list, y_pred))

def save_model(model, scaler, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving model to {output_path}...")
    joblib.dump({"model": model, "scaler": scaler}, output_path)
    print("Model saved successfully.")

# Main script
if __name__ == "__main__":
    images, labels = load_data(dataset_path)
    images_p, labels_p = preprocess_data(images, labels)
    X_train, X_test, y_train, y_test = split_data(images_p, labels_p)
    model, scaler = train_model(X_train, y_train)
    if model and scaler:
        evaluate_model(model, scaler, X_test, y_test)
    else:
        print("Model training failed. Cannot proceed with evaluation.")

    save_model(model, scaler, model_output_path)
