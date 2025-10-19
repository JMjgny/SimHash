import numpy as np
from saliency import compute_saliency_map
from segmentation import segment_image_regions, compute_regional_phash
from config import HASH_SIZE, NUM_REGIONS

def compute_weighted_simhash(image, num_regions=NUM_REGIONS):
    """Compute weighted SimHash from regional perceptual hashes"""
    saliency_map = compute_saliency_map(image)
    regions, weights = segment_image_regions(image, saliency_map, num_regions)
    
    regional_hashes = []
    for region in regions:
        phash = compute_regional_phash(region)
        regional_hashes.append(phash)
    
    final_hash = np.zeros(HASH_SIZE)
    
    for phash, weight in zip(regional_hashes, weights):
        hash_array = np.array([int(b) for b in bin(int(str(phash), 16))[2:].zfill(HASH_SIZE)])
        final_hash += hash_array * weight
    
    final_hash = (final_hash > 0.5).astype(int)
    hash_value = int(''.join(map(str, final_hash)), 2)
    
    return hash_value, regional_hashes, weights

def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two hash values"""
    return bin(hash1 ^ hash2).count('1')
