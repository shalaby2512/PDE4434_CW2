import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from scan import scan, average_color, reduce_overexposure
import os
import cv2
import numpy as np

# Get labels from folder names
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Wild', 'draw four', 'draw two', 'reverse', 'skip']
label_map = {label: idx for idx, label in enumerate(labels)}



def train_card_value_classifier(DATASET_PATH, model_save_path, IMG_SIZE=100, epochs=50):
    # Load and preprocess data
    X, y = [], []
    for label in labels:
        folder = os.path.join(DATASET_PATH, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label_map[label])

    X = np.array(X, dtype="float32") / 255.0  # normalize
    y = tf.keras.utils.to_categorical(y, num_classes=len(labels))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(labels), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    # Save model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")



def predict_card_value(img_path, model_path):   
    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(img_path)
    scanned = scan(img)

    if scanned is not None:
        #print(average_color(scanned))
        if average_color(scanned)[0] >= 100 and average_color(scanned)[1] >= 100 and average_color(scanned)[2] >= 100:
            exposure_reduced = reduce_overexposure(scanned)
            img = scanned
            img = cv2.resize(img, (100, 100))
            img = np.expand_dims(img / 255.0, axis=0)
            prediction = model.predict(img)
            predicted_label = labels[np.argmax(prediction)]
            return predicted_label
        else:
            img = cv2.resize(img, (100, 100))
            img = np.expand_dims(img / 255.0, axis=0)
            prediction = model.predict(img)
            predicted_label = labels[np.argmax(prediction)]
            return predicted_label
    else:
        print("image not scanned")
        # img = cv2.resize(img, (100, 100))
        # img = np.expand_dims(img / 255.0, axis=0)
        # prediction = model.predict(img)
        # predicted_label = labels[np.argmax(prediction)]
        # return predicted_label





# def test_folder(folder_path):
#     for path in os.listdir(folder_path):
#         p = os.path.join(folder_path,path)
#         print(f"image: {path} - detected value: {predict_card_value(p)}")

#test_folder("D:/Upwork/uno3/test2")