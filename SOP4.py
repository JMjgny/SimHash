import cv2
import numpy as np
from skimage.feature import hog
import hashlib

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

# Load images
original = cv2.imread("DataSet/Authentic/beforephoto.jpg")
modified = cv2.imread("DataSet/Fraud/blurface.jpg")

if original is None or modified is None:
    print("[ERROR] Could not load one or both images.")
    exit()

# Step 1: Compute hashes for full images
original_hash = simhash(image_to_feature_vector(original))
modified_hash = simhash(image_to_feature_vector(modified))
dist_full = hamming_distance(original_hash, modified_hash)

# Step 2: Resize image for ROI selection (max width or height = 800)
max_dim = 800
height, width = modified.shape[:2]
scale = 1.0

if max(height, width) > max_dim:
    scale = max_dim / max(height, width)
    resized = cv2.resize(modified, (int(width * scale), int(height * scale)))
else:
    resized = modified.copy()

print("Please select the tampered patch region on the resized image, then press ENTER or SPACE.")
roi = cv2.selectROI("Select Patch Region", resized)
cv2.destroyWindow("Select Patch Region")

# Scale ROI back to original image size
x, y, w, h = roi
x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

# Extract patch from original-size modified image
blank = np.zeros_like(modified)
blank[y:y+h, x:x+w] = modified[y:y+h, x:x+w]

# Save isolated patch
cv2.imwrite("isolated_patch.jpg", blank)

# Step 3: Compute hash of isolated patch
patch_hash = simhash(image_to_feature_vector(blank))
dist_patch = hamming_distance(original_hash, patch_hash)

# Report
print("Original SimHash:        ", bin(original_hash))
print("Modified Full SimHash:   ", bin(modified_hash))
print("Isolated Patch SimHash:  ", bin(patch_hash))
print(f"\nHamming Distance (Full Image):     {dist_full}")
print(f"Hamming Distance (Isolated Patch): {dist_patch}")
