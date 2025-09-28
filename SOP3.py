import cv2
import numpy as np
from skimage.feature import hog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hashlib

# --- SimHash core functions ---

def image_to_feature_vector(img):
    # Resize and convert to grayscale
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # HOG features simulate local feature extraction
    features, _ = hog(gray, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=True, feature_vector=True)
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

# --- Load and compare images ---

image_paths = {
    "Original": "DataSet/Authentic/beforephoto.jpg",
    "Modified": "DataSet/Fraud/blurface.jpg"
}

hashes = {}

for label, path in image_paths.items():
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Could not load image: {path}")
        continue

    features = image_to_feature_vector(img)
    image_hash = simhash(features)
    hashes[label] = (image_hash, img)

# --- Report results ---

original_hash = hashes["Original"][0]

for label in hashes:
    if label == "Original":
        print(f"{label} SimHash: {bin(original_hash)}")
        continue
    h = hashes[label][0]
    dist = hamming_distance(original_hash, h)
    print(f"\n{label} SimHash: {bin(h)}")
    print(f"Hamming Distance from Original: {dist}")

# --- Visualize ---
plt.figure(figsize=(10, 5))
for i, (label, (_, img)) in enumerate(hashes.items()):
    plt.subplot(1, len(hashes), i + 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(label)
    plt.axis('off')

plt.tight_layout()


