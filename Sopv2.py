import cv2
import numpy as np
from skimage.feature import hog
import hashlib
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid issues
import matplotlib.pyplot as plt

def image_to_feature_vector(img):
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return features

def simhash(features):
    bits = 64
    v = [0] * bits
    for i, val in enumerate(features):
        h = int(hashlib.md5(str(i).encode()).hexdigest(), 16)
        for j in range(bits):
            bitmask = 1 << j
            v[j] += val if h & bitmask else -val
    fingerprint = 0
    for i in range(bits):
        if v[i] >= 0:
            fingerprint |= 1 << i
    return fingerprint

def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')

def run_sop2_result(original_img_path, modified_img_path):
    # Load images, check existence
    if not os.path.exists(original_img_path):
        raise FileNotFoundError(f"Original image not found: {original_img_path}")
    if not os.path.exists(modified_img_path):
        raise FileNotFoundError(f"Test image not found: {modified_img_path}")

    original_img = cv2.imread(original_img_path)
    modified_img = cv2.imread(modified_img_path)
    if original_img is None or modified_img is None:
        raise FileNotFoundError("Error loading images.")

    # Compute SimHash for each
    original_hash = simhash(image_to_feature_vector(original_img))
    modified_hash = simhash(image_to_feature_vector(modified_img))
    dist = hamming_distance(original_hash, modified_hash)

    # Optional: visualization (if needed outside this function)
    # plt.subplot(121)
    # plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    # plt.title("Original")
    # plt.subplot(122)
    # plt.imshow(cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB))
    # plt.title("Modified")
    # plt.show()

    # Return formatted string result to match thesis output style
    return (f"Original SimHash: {bin(original_hash)}\n"
            f"Modified SimHash: {bin(modified_hash)}\n"
            f"Hamming Distance: {dist}")
