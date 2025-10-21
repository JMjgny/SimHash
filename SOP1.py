import cv2
import numpy as np
from hashlib import md5

def image_to_patches(img, patch_size=16):
    patches = []
    h, w = img.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

def patch_md5(patch):
    return md5(patch.tobytes()).hexdigest()

def run_sop1_result(original_img_path, modified_img_path):
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        raise FileNotFoundError(f"Original image not found: {original_img_path}")

    modified_img = cv2.imread(modified_img_path)
    if modified_img is None:
        raise FileNotFoundError(f"Modified image not found: {modified_img_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)

    patch_size = 16
    orig_patches = image_to_patches(original_img, patch_size)
    mod_patches = image_to_patches(modified_img, patch_size)

    total_patches = len(orig_patches)
    diff_count = 0

    for orig_patch, mod_patch in zip(orig_patches, mod_patches):
        if patch_md5(orig_patch) != patch_md5(mod_patch):
            diff_count += 1

    percent_diff = (diff_count / total_patches) * 100

    return (f"Total patches: {total_patches}\n"
            f"Number of patches with different MD5 hashes: {diff_count}\n"
            f"Percentage difference: {percent_diff:.2f}%")

# Optional: Example usage
if __name__ == "__main__":
    print(run_sop1_result(
        r"C:\Users\Marc\.cache\kagglehub\datasets\labid93\image-forgery-detection\versions\1\Dataset\Original\before_photo.jpg",
        r"C:\Users\Marc\.cache\kagglehub\datasets\labid93\image-forgery-detection\versions\1\Dataset\Forged\bright_blonde.jpg"
    ))
