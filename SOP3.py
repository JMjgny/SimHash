import cv2
import numpy as np
from skimage.feature import hog
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def image_to_feature_vector(img):
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
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

def run_sop3_result(original_img_path, modified_img_path):
    image_paths = {"Original": original_img_path, "Modified": modified_img_path}
    hashes = {}

    for label, path in image_paths.items():
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"[ERROR] Could not load image: {path}")
        features = image_to_feature_vector(img)
        image_hash = simhash(features)
        hashes[label] = (image_hash, img)

    original_hash = hashes["Original"][0]

    results = []
    for label in hashes:
        if label == "Original":
            results.append(f"{label} SimHash: {bin(original_hash)}")
            continue
        h = hashes[label][0]
        dist = hamming_distance(original_hash, h)
        results.append(f"{label} SimHash: {bin(h)}")
        results.append(f"Hamming Distance from Original: {dist}")

    # Optionally, save or plot images here if you want in your function.

    return "\n".join(results)
