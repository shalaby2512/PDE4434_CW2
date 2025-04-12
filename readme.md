
# 🧾 Project Report: UNO Card Detection System  
**Objective**:  
To detect and classify UNO cards by both their **color** and **value** using computer vision techniques and machine learning models.


GitHub Repo: https://github.com/shalaby2512/PDE4434_CW2

Data Set Google Drive Link: https://drive.google.com/drive/folders/11yN3zVfLBTUn9ubSIzHpqqGVRumpxViq?usp=share_link


---

## 1. 📸 Data Collection

### Overview:
Two types of datasets were created:
- **Color classification dataset**: Images of UNO cards categorized by **color** (red, blue, green, yellow).
- **Value classification dataset**: Images categorized by **card value** (e.g., 0–9, skip, reverse, draw two, draw four, wild).

### Requirements:
- Well-lit, high-resolution photos of cards.
- Cards placed against a clear, contrasting background.
- Each category stored in a separate folder:
  - For colors: `original/red/`, `original/blue/`, etc.
  - For values: `original/0/`, `original/reverse/`, etc.

---

## 2. 🧽 Scanning & Preprocessing

### Module: `scan.py`

Each image is preprocessed before being used for model training. The process includes:
- **Grayscale conversion** and **blurring** to reduce noise.
- **Canny edge detection** to find outlines.
- **Contour detection** to isolate the card shape.
- **Perspective transformation** to get a flat, rectangular top-down view of the card.
- **Resizing** the image for uniformity (typically 100x100 pixels).
- **Average brightness** is computed; if brightness is too high (overexposed), color is adjusted using the LAB color space.

### Function: `preprocess_raw_images(path)`
- Automatically reads and processes all images in the provided path.
- Saves the processed images in the `preprocessed/` directory with the same folder structure.

- If scan of any image is failed, it is not included in the preprocess folder

---

## 3. 🎨 Color Classification Model

### Module: `color_det.py`

### Model Used:
- **RandomForestClassifier** from `scikit-learn`.

### Features:
- Average RGB color values from the preprocessed card image.
- These 3-dimensional values (R, G, B) are used as input to the model.

### Training Function:
```python
train_color_classifier(data_path, model_output_path)
```

### Output:
- Saves the trained model as `card_color_model.pkl`.

---

## 4. 🔢 Value Classification Model

### Module: `num_detect.py`

### Model Used:
- A **Convolutional Neural Network (CNN)** built using **TensorFlow/Keras**.

### Input:
- The entire resized RGB image (100x100x3) of the card face.

### Architecture:
- 2 Convolutional + MaxPooling layers.
- Flatten + Dense layers for classification.
- Output layer uses softmax for multiclass prediction.

### Training Function:
```python
train_card_value_classifier(data_path, model_output_path)
```

### Output:
- Model saved as `card_value_classifier.h5`.

---

## 5. 🧪 Testing

### Module: `main.py`

### Functions:
- `test_image(path)`: Classifies one image.
- `test_folder(folder_path)`: Tests all images in a folder.

### Output:
For each test image, prints a result like:
```
image: card3.jpg - it is a green 8
image: card5.jpg - it is a draw four
```

---

## 6. 📊 Evaluation Metrics

### Color Classifier:
Evaluated using `classification_report()` from scikit-learn which provides:
- **Precision**: Accuracy of predicted labels.
- **Recall**: Coverage of actual labels.
- **F1-score**: Harmonic mean of precision and recall.

Example:
```
              precision    recall  f1-score   support

        blue       1.00      1.00      1.00         7
       green       1.00      1.00      1.00         4
         red       1.00      1.00      1.00         9
      yellow       1.00      1.00      1.00         8

    accuracy                           1.00        28
   macro avg       1.00      1.00      1.00        28
weighted avg       1.00      1.00      1.00        28
```

### Value Classifier:
Evaluated via validation split during training.

Metrics:
- **Training Accuracy**
- **Validation Accuracy**
- **Loss** graphs plotted during training.

Example:
```
95ms/step - accuracy: 1.0000 - loss: 3.7530e-04 - val_accuracy: 0.9697 - val_loss: 0.1491
```

---

## 7. 🛠 Tools & Libraries Used

| Purpose                | Library        |
|------------------------|----------------|
| Image processing       | OpenCV         |
| Numerical computation  | NumPy          |
| Preprocessing          | imutils        |
| Machine Learning       | scikit-learn   |
| Deep Learning          | TensorFlow     |
| Model Serialization    | joblib, H5     |

---

## 8. ✅ Conclusion

The UNO card detection system effectively uses a hybrid approach:
- A **Random Forest** classifier based on color statistics.
- A **CNN** for visual pattern recognition of card values.
- Preprocessing ensures uniform, high-quality input for both models.

This makes the solution robust and extendable. Additional cards, effects, or colors can be added by retraining the system with new data.
