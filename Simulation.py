import cv2 as cv
import numpy as np
from simhash import Simhash
import torch
import torchvision.transforms as transforms
from torchvision import models
import pywt
import os
from torchvision.models import EfficientNet_B0_Weights

# Load a pre-trained AI detection model (EfficientNet-B0 for feature extraction)
ai_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
ai_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def image_to_feature_vector(image_path, size=(64, 64)):
    """Convert an image to a feature vector by extracting color, shape, texture, and frequency information."""
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read image {image_path}")
    image = cv.resize(image, size)
    
    # Extract edge features using Canny edge detection
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    
    # Extract high-frequency texture details
    laplacian = cv.Laplacian(gray, cv.CV_64F).var()
    noise = np.std(gray)
    
    # Compute color statistics
    mean_color = np.mean(image, axis=(0, 1))
    color_std_dev = np.std(image, axis=(0, 1))
    
    # Frequency domain analysis using Discrete Wavelet Transform (DWT)
    coeffs2 = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs2
    wavelet_features = np.array([np.mean(np.abs(cH)), np.mean(np.abs(cV)), np.mean(np.abs(cD))])
    
    # Deep Learning Feature Extraction
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        deep_features = ai_model(image_tensor).flatten().numpy()
    
    return np.concatenate((edges.flatten(), wavelet_features, deep_features))

def compute_simhash(image_path):
    """Compute the SimHash of an image based on its feature vector."""
    feature_vector = image_to_feature_vector(image_path)
    feature_groups = [str(int(x)) for x in feature_vector[:2048]]
    return Simhash(feature_groups, f=128)

def hamming_distance(hash1, hash2):
    """Compute the Hamming distance between two SimHash values."""
    return hash1.distance(hash2)

def detect_forgery(image_path, dataset_folder, threshold=15):
    """Compare the image's SimHash with a dataset of original images to detect splicing or copy-move forgery."""
    image_hash = compute_simhash(image_path)
    
    similar_images = []
    
    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        if os.path.isfile(file_path):
            dataset_hash = compute_simhash(file_path)
            distance = hamming_distance(image_hash, dataset_hash)
            
            if distance <= threshold:
                similar_images.append((file, distance))
    
    if similar_images:
        print(f"Possible forgery detected! Similar images found:")
        for img, dist in similar_images:
            print(f" - {img} (Hamming Distance: {dist})")
    else:
        print("No significant signs of forgery detected.")

def main():
    image1 = "forged/12731.jpg"  # Replace with actual image path
    dataset_folder = "original"  # Folder containing reference/original images
    
    try:
        detect_forgery(image1, dataset_folder)
    except FileNotFoundError as e:
        print(e)
        return

if __name__ == "__main__":
    main()
    