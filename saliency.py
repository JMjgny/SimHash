import cv2 as cv
import numpy as np

def compute_saliency_map(image):
    """Compute saliency map using spectral residual method"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    saliency = cv.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    
    if not success:
        grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        saliency_map = np.sqrt(grad_x**2 + grad_y**2)
        saliency_map = (saliency_map * 255 / np.max(saliency_map)).astype(np.uint8)
    else:
        saliency_map = (saliency_map * 255).astype(np.uint8)
    
    return saliency_map
