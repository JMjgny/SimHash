import os
import cv2 as cv
import hashlib
import numpy as np
import pandas as pd
import kagglehub

def image_to_patches(image, patch_size=(8, 8)):
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h, patch_size[1]):
        for x in range(0, w, patch_size[0]):
            patch = image[y:y+patch_size[1], x:x+patch_size[0]]
            if patch.shape[0] == patch_size[1] and patch.shape[1] == patch_size[0]:
                patches.append(patch)
    return patches

def md5_hash_patch(patch):
    patch_flat = patch.flatten().tobytes()
    return hashlib.md5(patch_flat).hexdigest()

def md5_to_bits(md5hash):
    return bin(int(md5hash, 16))[2:].zfill(128)

def simhash_md5_frequency(image, patch_size=(8, 8)):
    patches = image_to_patches(image, patch_size)
    hashes = [md5_to_bits(md5_hash_patch(p)) for p in patches]
    freq = {}
    for h in hashes:
        freq[h] = freq.get(h, 0) + 1
    weights = [freq[h] / len(hashes) for h in hashes]
    hash_matrix = np.array([[int(bit) for bit in h] for h in hashes])
    weighted_sum = np.dot(weights, hash_matrix)
    simhash = (weighted_sum > 0.5).astype(int)
    return simhash

def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def scan_dataset_create_csv(originals_folder, forged_folder, csv_output_path):
    data = []
    forged_files = os.listdir(forged_folder)
    for forged_file in forged_files:
        forged_path = os.path.join(forged_folder, forged_file)
        forged_img = cv.imread(forged_path, cv.IMREAD_GRAYSCALE)
        forged_hash = simhash_md5_frequency(forged_img)

        best_match = None
        best_dist = float('inf')

        for orig_file in os.listdir(originals_folder):
            orig_path = os.path.join(originals_folder, orig_file)
            orig_img = cv.imread(orig_path, cv.IMREAD_GRAYSCALE)
            orig_hash = simhash_md5_frequency(orig_img)
            dist = hamming_distance(forged_hash, orig_hash)

            if dist < best_dist:
                best_dist = dist
                best_match = orig_file

        data.append({
            'Forged Image': forged_file,
            'Best Matched Original': best_match,
            'Hamming Distance': best_dist
        })
        print(f"Processed forged image {forged_file}: Best original match={best_match} with Hamming distance={best_dist}")

    df = pd.DataFrame(data)
    df.to_csv(csv_output_path, index=False)
    return df

if __name__ == '__main__':
    dataset_root = kagglehub.dataset_download("labid93/image-forgery-detection")
    original_folder = os.path.join(dataset_root, "DataSet/Original")
    forged_folder = os.path.join(dataset_root, "DataSet/Forged")
    output_csv = "forgery_match_results.csv"

    df_results = scan_dataset_create_csv(original_folder, forged_folder, output_csv)
    print(f"\nCSV file '{output_csv}' created with forgery detection matches.")
    print(df_results.head())
