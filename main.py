"""
main.py
---------
Main script to train, preprocess, and test UNO card classifier.
"""

import os
from color_det import train_color_classifier, predict_card_color
from num_detect import train_card_value_classifier, predict_card_value
from scan import preprocess_dataset

preprocess_data = False
train_model = False #make this true if you want to train the model
test_model = True #make this true if you want to test the model

ORIGINAL_PATH = "original"
DATASET_PATH = "preprocessed"
COLOR_MODEL_PATH = "card_color_model.pkl"
VALUE_MODEL_PATH = "card_value_classifier.h5"
test_folder_path = "test"

def preprocess_raw_images(ORIGINAL_PATH):
    """
    Scans and saves all images from raw dataset into usable format.
    """

    preprocess_dataset(ORIGINAL_PATH, DATASET_PATH)

def train():
    """
    Trains both color and value classifiers.
    """

    train_color_classifier(DATASET_PATH, COLOR_MODEL_PATH)
    train_card_value_classifier(DATASET_PATH, VALUE_MODEL_PATH)


def test_folder(folder_path):
    """
    Tests all images in a folder and returns predictions.

    Args:
        folder_path (str): Path to folder with test images.

    Returns:
        list: List of string results.
    """
    results = []
    for path in os.listdir(folder_path):
        p = os.path.join(folder_path,path)
        # print(f"image: {path} - detected value: {predict_card_value(p)}")
        # print(f"image: {path} - detected colour: {predict_card_color(p)}")
        value = predict_card_value(p, VALUE_MODEL_PATH)
        if value == "wild" or value == "draw four":
            #print(f"image: {path} - it is a {value}")
            results.append(f"image: {path} - {value}")
        else:
            color = predict_card_color(p, COLOR_MODEL_PATH)      
            #print(f"image: {path} - it is a {color} {value}")
            results.append(f"image: {path} -{color} {value}")
    return results

def test_image(image_path):
    """
    Predicts value and color of a single image.

    Args:
        image_path (str): Path to image file.
    """

    color = predict_card_color(image_path, COLOR_MODEL_PATH)
    value = predict_card_value(image_path, VALUE_MODEL_PATH)
    print(f"image: {image_path} - it is a {color} {value}")


if preprocess_data:
    preprocess_raw_images(ORIGINAL_PATH)
if train_model:
    train()
if test_model:
    results = test_folder(test_folder_path)
    for r in results:
        print(r)