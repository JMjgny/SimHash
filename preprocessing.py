import os
import cv2 as cv
from config import (IMAGE_SIZE, USE_FULL_PREPROCESSING, USE_CLAHE, 
                    USE_DENOISING, USE_SHARPENING, CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE)

def preprocess_image(image_path, size=IMAGE_SIZE):
    """
    Enhanced preprocessing with configurable options
    Set USE_FULL_PREPROCESSING=True in config.py for complete algorithm
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image not found {image_path}")
    
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read image {image_path}")
    
    # Apply preprocessing based on config
    if USE_FULL_PREPROCESSING or USE_CLAHE:
        # Convert to LAB color space for CLAHE
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        l = clahe.apply(l)
        
        # Merge back and convert to BGR
        lab = cv.merge([l, a, b])
        image = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    
    if USE_FULL_PREPROCESSING or USE_DENOISING:
        # Apply Non-Local Means Denoising (SLOWEST OPERATION)
        image = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    if USE_FULL_PREPROCESSING or USE_SHARPENING:
        # Unsharp masking for edge sharpening
        gaussian = cv.GaussianBlur(image, (0, 0), 2.0)
        image = cv.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    # Resize to standard size
    image = cv.resize(image, size)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    return image_rgb, image
