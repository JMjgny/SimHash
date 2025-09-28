import cv2
import numpy as np
from PIL import Image
from hashlib import md5
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid TclError
import matplotlib.pyplot as plt
import os

# --- SimHash Implementation ---
def simhash(features):
    hashbits = 64
    v = [0] * hashbits
    for feature in features:
        h = int(md5(feature.encode('utf-8')).hexdigest(), 16)
        for i in range(hashbits):
            bitmask = 1 << i
            v[i] += 1 if h & bitmask else -1
    fingerprint = 0
    for i in range(hashbits):
        if v[i] >= 0:
            fingerprint |= 1 << i
    return fingerprint

def hamming_distance(x, y):
    return bin(x ^ y).count('1')

# --- Feature Extraction ---
def extract_features(img):
    resized = cv2.resize(img, (32, 32))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = []
    for row in gray:
        features.extend([str(p) for p in row])
    return features

# --- Image Paths ---
image_paths = {
    "Original": "DataSet/Authentic/Harry.jpg",
    "Lighting Increased": "DataSet/Fraud/Harry_Bright.jpeg",
    "Noise": "DataSet/Fraud/Harry_Noise.jpg",
    "Blurred": "DataSet/Fraud/Harry_Blur.jpg"
}

# --- Load Images and Compute Hashes ---
images = {}
hashes = {}

for label, path in image_paths.items():
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        continue
    img = cv2.imread(path)
    images[label] = img
    features = extract_features(img)
    hashes[label] = simhash(features)

# --- Compare Hamming Distances ---
original_hash = hashes.get("Original")
if original_hash is None:
    print("Original image not loaded. Cannot proceed.")
else:
    print(f"\nOriginal SimHash: {bin(original_hash)}\n")
    for label, hash_val in hashes.items():
        if label == "Original":
            continue
        dist = hamming_distance(original_hash, hash_val)
        print(f"{label} SimHash: {bin(hash_val)}")
        print(f"Hamming Distance from Original: {dist}\n")

# --- Plot Comparison Images ---
num_images = len(images)
fig, axs = plt.subplots(1, num_images, figsize=(4 * num_images, 4))

# Ensure axs is iterable even if only one subplot
if num_images == 1:
    axs = [axs]

for idx, (label, img) in enumerate(images.items()):
    axs[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[idx].set_title(label)
    axs[idx].axis('off')

plt.tight_layout()

