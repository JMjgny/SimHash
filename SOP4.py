import cv2
import numpy as np
from skimage.feature import hog
import hashlib

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

def run_sop4_result(original_img_path, modified_img_path, isolated_patch_path=None):
    # Load images
    orig_img = cv2.imread(original_img_path)
    mod_img = cv2.imread(modified_img_path)
    if orig_img is None or mod_img is None:
        raise FileNotFoundError("Original or modified image not found.")

    # Step 1: Compute full image hashes
    orig_hash = simhash(image_to_feature_vector(orig_img))
    mod_hash = simhash(image_to_feature_vector(mod_img))
    dist_full = hamming_distance(orig_hash, mod_hash)

    # Step 2: If isolated_patch_path is not given, allow interactive ROI selection
    if isolated_patch_path is None:
        max_dim = 800
        height, width = mod_img.shape[:2]
        scale = 1.0
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            resized = cv2.resize(mod_img, (int(width * scale), int(height * scale)))
        else:
            resized = mod_img.copy()

        print("Select tampered patch region, then press ENTER or SPACE.")
        roi = cv2.selectROI("Select Patch Region", resized)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        isolated_patch = np.zeros_like(mod_img)
        isolated_patch[y:y+h, x:x+w] = mod_img[y:y+h, x:x+w]
    else:
        isolated_patch = cv2.imread(isolated_patch_path)
        if isolated_patch is None:
            raise FileNotFoundError(f"Isolated patch image not found: {isolated_patch_path}")

    # Step 3: Compute isolated patch hash
    patch_hash = simhash(image_to_feature_vector(isolated_patch))
    dist_patch = hamming_distance(orig_hash, patch_hash)

    # Prepare output string for comparison
    return (
        f"Original SimHash:        {bin(orig_hash)}\n"
        f"Modified Full SimHash:    {bin(mod_hash)}\n"
        f"Isolated Patch SimHash:   {bin(patch_hash)}\n\n"
        f"Hamming Distance (Full Image):     {dist_full}\n"
        f"Hamming Distance (Isolated Patch): {dist_patch}"
    )
