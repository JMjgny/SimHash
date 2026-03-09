import cv2 as cv
import numpy as np
from simhash import Simhash
import os
import time
import pickle

PRECOMPUTED_HASHES_FILE = "precomputed_hashes.pkl"
PATCH_SIZE = 8

def image_to_feature_vector_simple(image_path, patch_size=PATCH_SIZE):
    """Convert an image to a feature vector using simple patches (pixel values)."""
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read image {image_path}. Check file path and integrity.")
    h, w = image.shape
    feature_groups = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size):
                feature_groups.append(''.join(str(x) for x in patch.flatten()))
    return feature_groups

def compute_simhash_simple(image_path, patch_size=PATCH_SIZE):
    """Compute the SimHash of an image using simple patches."""
    feature_groups = image_to_feature_vector_simple(image_path, patch_size)
    return Simhash(feature_groups, f=128)

def precompute_hashes(original_dir, patch_size=PATCH_SIZE):
    """Precompute and save SimHashes for all original images."""
    hashes = {}
    for filename in os.listdir(original_dir):
        file_path = os.path.join(original_dir, filename)
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        try:
            simhash = compute_simhash_simple(file_path, patch_size)
            hashes[file_path] = simhash
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    with open(PRECOMPUTED_HASHES_FILE, 'wb') as f:
        pickle.dump(hashes, f)
    print(f"Precomputed hashes for {len(hashes)} images and saved to {PRECOMPUTED_HASHES_FILE}")

def load_precomputed_hashes():
    """Load precomputed hashes from file."""
    if os.path.exists(PRECOMPUTED_HASHES_FILE):
        with open(PRECOMPUTED_HASHES_FILE, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Precomputed hashes file not found: {PRECOMPUTED_HASHES_FILE}")

def find_original_image(forged_image_path, hashes, patch_size=PATCH_SIZE):
    """Find the most similar (original) image in the dataset to a forged image."""
    try:
        forged_simhash = compute_simhash_simple(forged_image_path, patch_size)
    except FileNotFoundError as e:
        print(e)
        return None, float('inf')
    
    best_match = None
    min_distance = float('inf')
    for file_path, original_simhash in hashes.items():
        distance = forged_simhash.distance(original_simhash)
        if distance < min_distance:
            min_distance = distance
            best_match = file_path
    return best_match, min_distance

def main():
    dataset_root = r"C:\Users\Marc\.cache\kagglehub\datasets\labid93\image-forgery-detection\versions\1\Dataset"
    forged_image = r"C:\Users\Marc\.cache\kagglehub\datasets\labid93\image-forgery-detection\versions\1\Dataset\Forged\11037.jpg" 

    original_dir = os.path.join(dataset_root, "original")
    if not os.path.exists(PRECOMPUTED_HASHES_FILE):
        print("Precomputing hashes for original images...")
        precompute_hashes(original_dir)
    
    try:
        start_total = time.time()
        hashes = load_precomputed_hashes()
        original_image, distance = find_original_image(forged_image, hashes)
        total_time = time.time() - start_total
        if original_image:
            print(f"Most similar original image: {original_image}")
            print(f"SimHash distance: {distance}")
            print(f"Total processing time: {total_time:.4f} seconds")
        else:
            print("No matching image found.")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
