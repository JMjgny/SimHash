import numpy as np
import cv2
from saliency import compute_saliency_map
from segmentation import segment_image_regions, compute_regional_phash
from config import HASH_SIZE, NUM_REGIONS


def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')


def _hex_phash_to_bits(phash_hex):
    """Convert hex pHash to a binary array of length HASH_SIZE."""
    return np.array(
        [int(b) for b in bin(int(str(phash_hex), 16))[2:].zfill(HASH_SIZE)],
        dtype=np.float32
    )


def compute_frequency_weights(image, regions):
    """
    Frequency-based weights:
    use grayscale variance as a simple texture/frequency proxy.
    """
    variances = []
    for region in regions:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        var = np.var(gray)
        variances.append(var)

    variances = np.array(variances, dtype=np.float32)

    if np.all(variances == 0):
        # fallback: uniform weights
        weights = np.ones_like(variances) / len(variances)
    else:
        weights = variances / np.sum(variances)

    return weights


def compute_weighted_hash(image, mode="saliency", num_regions=NUM_REGIONS):
    """
    mode = "saliency"  → Adaptive Semantic Saliency Weighting
    mode = "frequency" → Frequency-based weighting (variance)
    Returns: hash_value (int)
    """

    if mode == "saliency":
        saliency_map = compute_saliency_map(image)
        regions, weights = segment_image_regions(image, saliency_map, num_regions)
    elif mode == "frequency":
        regions, _ = segment_image_regions(image, None, num_regions)
        weights = compute_frequency_weights(image, regions)
    else:
        raise ValueError("mode must be 'saliency' or 'frequency'")

    regional_hashes = []
    for region in regions:
        phash = compute_regional_phash(region)  # hex string
        regional_hashes.append(phash)

    final_hash_bits = np.zeros(HASH_SIZE, dtype=np.float32)

    for phash, weight in zip(regional_hashes, weights):
        hash_array = _hex_phash_to_bits(phash)
        final_hash_bits += hash_array * weight

    final_hash_bits = (final_hash_bits > 0.5).astype(np.int32)
    hash_value = int("".join(map(str, final_hash_bits)), 2)

    return hash_value


def compare_frequency_vs_saliency(original_path, forged_path):
    """
    Compute and compare:
      - frequency-based weighted hash
      - saliency-based weighted hash
    for original vs forged image.

    Returns a dictionary with the Hamming distances.
    """
    orig = cv2.imread(original_path)
    forged = cv2.imread(forged_path)

    if orig is None:
        raise FileNotFoundError(f"Could not load original image: {original_path}")
    if forged is None:
        raise FileNotFoundError(f"Could not load forged image: {forged_path}")

    orig_freq_hash = compute_weighted_hash(orig, mode="frequency")
    forged_freq_hash = compute_weighted_hash(forged, mode="frequency")

    orig_sal_hash = compute_weighted_hash(orig, mode="saliency")
    forged_sal_hash = compute_weighted_hash(forged, mode="saliency")


    freq_dist = hamming_distance(orig_freq_hash, forged_freq_hash)
    sal_dist = hamming_distance(orig_sal_hash, forged_sal_hash)

    results = {
        "frequency_based_hamming": freq_dist,
        "saliency_based_hamming": sal_dist
    }

    print("Frequency-based Hamming distance (orig vs forged):", freq_dist)
    print("Saliency-based Hamming distance  (orig vs forged):", sal_dist)

    return results
