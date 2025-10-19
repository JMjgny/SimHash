import cv2 as cv
import numpy as np
import imagehash
from PIL import Image
from config import NUM_REGIONS

def segment_image_regions(image, saliency_map, num_regions=NUM_REGIONS):
    """Divide image into regions based on grid"""
    h, w = image.shape[:2]
    grid_h = int(np.sqrt(num_regions))
    grid_w = num_regions // grid_h
    
    region_h = h // grid_h
    region_w = w // grid_w
    
    regions = []
    region_weights = []
    
    for i in range(grid_h):
        for j in range(grid_w):
            y1, y2 = i * region_h, (i + 1) * region_h
            x1, x2 = j * region_w, (j + 1) * region_w
            
            region = image[y1:y2, x1:x2]
            saliency_region = saliency_map[y1:y2, x1:x2]
            weight = np.mean(saliency_region) / 255.0
            
            regions.append(region)
            region_weights.append(weight)
    
    total_weight = sum(region_weights)
    region_weights = [w / total_weight for w in region_weights]
    
    return regions, region_weights

def compute_regional_phash(region):
    """Compute perceptual hash for a region"""
    region_pil = Image.fromarray(cv.cvtColor(region, cv.COLOR_BGR2RGB))
    phash = imagehash.phash(region_pil, hash_size=8)
    return phash
