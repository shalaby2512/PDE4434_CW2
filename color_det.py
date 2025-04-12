"""
color_det.py
-------------
Handles training and prediction of UNO card colors using Random Forest.
"""

import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scan import scan, average_color, reduce_overexposure

# Constants

CLASSES = ['blue', 'yellow', 'green', 'red']


def extract_features(dataset_path):

    """
    Extracts average BGR color features from dataset images.

    Args:
        dataset_path (str): Path to dataset.

    Returns:
        tuple: Feature matrix (X), label vector (y)
    """

    X = []
    y = []
    for label in CLASSES:
        folder = os.path.join(dataset_path, label)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))
            avg_color = cv2.mean(img)[:3]
            X.append(avg_color)
            y.append(label)
    return np.array(X), np.array(y)


def train_color_classifier(DATASET_PATH, MODEL_PATH):
    """
    Trains a RandomForest model for color classification.

    Args:
        DATASET_PATH (str): Path to dataset.
        MODEL_PATH (str): Path to save the model.
    """

    X, y = extract_features(DATASET_PATH)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")



def predict_card_color(image_path, MODEL_PATH):
    """
    Predicts the color of an UNO card from an image.

    Args:
        image_path (str): Path to card image.
        MODEL_PATH (str): Path to the trained model.

    Returns:
        str: Predicted card color.
    """
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please run train() first.")

    model = joblib.load(MODEL_PATH)

    img = cv2.imread(image_path)
    if img is None:
        print("no image at path")
        return
    scanned = scan(img)

    if scanned is not None:
        #print(average_color(scanned))
        if average_color(scanned)[0] >= 100 and average_color(scanned)[1] >= 100 and average_color(scanned)[2] >= 100:
            exposure_reduced = reduce_overexposure(scanned)
            img = scanned
            img = cv2.resize(img, (100, 100))
            avg_color = cv2.mean(img)[:3]
            return model.predict([avg_color])[0]
        else:
            img = cv2.resize(img, (100, 100))
            avg_color = cv2.mean(img)[:3]
            return model.predict([avg_color])[0]
    else:
        print("image not scanned")
        # img = cv2.resize(img, (100, 100))
        # avg_color = cv2.mean(img)[:3]
        # return model.predict([avg_color])[0]


# def test_folder(folder_path):
#     for path in os.listdir(folder_path):
#         p = os.path.join(folder_path,path)
#         print(f"image: {path} - detected colour: {predict_card_color(p)}")
         

#train()
#test_folder("D:/Upwork/uno3/test")