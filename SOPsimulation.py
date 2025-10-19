import cv2
import numpy as np
import hashlib
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

def image_to_tokens(image):
    """Splits an image into pixel-based tokens (ineffective for meaningful feature extraction)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [gray[i, j] for i in range(gray.shape[0]) for j in range(gray.shape[1])], gray

def md5_hash(value):
    """Computes MD5 hash of a given pixel value."""
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16)

def compute_simhash(tokens):
    """Simulates SimHash with term frequency weighting (ineffective for images)."""
    vector = np.zeros(128)
    token_weights = defaultdict(int)
    
    # Frequency-based weighting (ineffective for images)
    for token in tokens:
        token_weights[token] += 1
    
    for token, weight in token_weights.items():
        hash_val = md5_hash(token)
        
        for i in range(128):
            if (hash_val >> i) & 1:
                vector[i] += weight
            else:
                vector[i] -= weight
    
    # Generate final SimHash (not preserving visual similarity)
    simhash = 0
    for i in range(128):
        if vector[i] >= 0:
            simhash |= (1 << i)
    
    return simhash

def compare_hashes(hash1, hash2):
    """Computes Hamming distance between two hashes."""
    return bin(hash1 ^ hash2).count('1')

def show_image_comparison(image1, image2, title1, title2):
    """Saves images for comparison instead of displaying them in a GUI."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    # Save the comparison image
    plt.savefig('image_comparison.png')
    plt.close()

# Load images
image_original = cv2.imread('DataSet/Authentic/566.jpg')
image_modified1 = cv2.imread('DataSet/Fraud/10995.jpg')

if image_original is None or image_modified1 is None:
    raise FileNotFoundError("One or more images not found!")

# Problem 1: Tokenization Issue
tokens_original, _ = image_to_tokens(image_original)
print(f"Problem 1: Tokenization - Extracted {len(tokens_original)} tokens")

# Problem 2: Hashing Issue (Show how small changes affect MD5 hash)
tokens_modified1, _ = image_to_tokens(image_modified1)
simhash_original = compute_simhash(tokens_original)
simhash_modified1 = compute_simhash(tokens_modified1)
print(f"Problem 2: Hashing Issue - SimHash Original: {simhash_original}")
print(f"Problem 2: Hashing Issue - SimHash Modified: {simhash_modified1}")

# Problem 3: Weighting Issue (SimHash uses equal weighting)
distance1 = compare_hashes(simhash_original, simhash_modified1)
print(f"Problem 3: Weighting Issue - Hamming Distance (Original vs Modified 1): {distance1}")

# Problem 4: Merging Issue (Compare copy-move forgery effects)
tokens_modified1, _ = image_to_tokens(image_modified1)
simhash_modified1 = compute_simhash(tokens_modified1)
distance2 = compare_hashes(simhash_original, simhash_modified1)
print(f"Problem 4: Merging Issue - SimHash Copy-Move Forgery: {simhash_modified1}")
print(f"Problem 4: Merging Issue - Hamming Distance (Original vs Modified 1): {distance2}")

# Save the image comparison instead of showing it
show_image_comparison(image_original, image_modified1, 'Original Image', 'Modified Image')
