"""
scan.py
--------
Provides scanning, preprocessing, and image transformation utilities.
"""

import cv2
import os
import numpy as np
import imutils

def reduce_overexposure(image):
    """
    Reduces brightness of overexposed images.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Brightness-adjusted image.
    """

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Clip the L channel to reduce overexposed brightness
    l = np.clip(l, 0, 200)  # cap max brightness (255 → 200)

    # Merge channels back
    lab = cv2.merge((l, a, b))

    # Convert back to BGR
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return adjusted

def average_color(img):
    """
    Calculates average color of image in RGB.

    Args:
        img (np.ndarray): Image.

    Returns:
        tuple: Average RGB color.
    """

    # Compute the average color per channel
    avg_bgr = cv2.mean(img)[:3]  # ignore alpha if present

    # Convert BGR to RGB
    avg_rgb = (avg_bgr[2], avg_bgr[1], avg_bgr[0])

    # print(f"Average RGB: {avg_rgb}")
    return avg_rgb

def order_points(pts):
    """
    Orders 4 corner points in consistent order: tl, tr, br, bl.

    Args:
        pts (np.ndarray): Corner points.

    Returns:
        np.ndarray: Ordered points.
    """

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """
    Applies a perspective transform to image.

    Args:
        image (np.ndarray): Input image.
        pts (np.ndarray): Corner points.

    Returns:
        np.ndarray: Warped image.
    """

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def scan(image):

    """
    Detects and scans a rectangular object (card) from the image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray or None: Scanned card if found, else None.
    """

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    if len(cnts) == 0:
        return None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        return None

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return warped

def preprocess_dataset(input_folder, output_folder):
    """
    Preprocesses entire dataset by scanning and saving valid card images.

    Args:
        input_folder (str): Raw dataset path.
        output_folder (str): Output folder for processed images.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    classes = os.listdir(input_folder)

    for label in classes:
        in_class_path = os.path.join(input_folder, label)
        out_class_path = os.path.join(output_folder, label)
        os.makedirs(out_class_path, exist_ok=True)

        for img_name in os.listdir(in_class_path):
            img_path = os.path.join(in_class_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"[!] Skipped unreadable image: {img_path}")
                continue

            scanned = scan(image)
            if scanned is not None:
                #print(average_color(scanned))
                if average_color(scanned)[0] >= 100 and average_color(scanned)[1] >= 100 and average_color(scanned)[2] >= 100:
                    out_path = os.path.join(out_class_path, img_name)
                    #exposure_reduced = reduce_overexposure(scanned)
                    cv2.imwrite(out_path, scanned)
                    print(f"[✓] Saved: {out_path}")
                else:
                    print(f"[x] Card not detected in: {img_path}")
            else:
                print(f"[x] Card not detected in: {img_path}")

