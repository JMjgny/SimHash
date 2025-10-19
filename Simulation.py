import os
import cv2 as cv
import numpy as np
import imagehash
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import shutil
import kagglehub

# ------------------ Step 1: Image Preprocessing ------------------
def preprocess_image(image_path, size=(224, 224)):
    """
    Enhanced preprocessing with CLAHE, Non-Local Means Denoising, and Unsharp Masking
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image not found {image_path}")
    
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read image {image_path}")
    
    # Convert to LAB color space for CLAHE
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back and convert to BGR
    lab = cv.merge([l, a, b])
    image = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    
    # Apply Non-Local Means Denoising
    image = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # Unsharp masking for edge sharpening
    gaussian = cv.GaussianBlur(image, (0, 0), 2.0)
    image = cv.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    # Resize to standard size
    image = cv.resize(image, size)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    return image_rgb, image

# ------------------ Step 2: Feature Extraction using ResNet-50 ------------------
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

# Initialize ResNet-50 feature extractor
resnet_extractor = ResNet50FeatureExtractor()
resnet_extractor.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_deep_features(image_rgb):
    """Extract high-level features using ResNet-50"""
    image_tensor = transform(image_rgb).unsqueeze(0)
    with torch.no_grad():
        features = resnet_extractor(image_tensor)
    return features.flatten().numpy()

# ------------------ Step 2 & 3: Region Segmentation and Perceptual Hashing ------------------
def compute_saliency_map(image):
    """Compute saliency map using spectral residual method"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Create saliency detector
    saliency = cv.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    
    if not success:
        # Fallback: simple gradient-based saliency
        grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        saliency_map = np.sqrt(grad_x**2 + grad_y**2)
        saliency_map = (saliency_map * 255 / np.max(saliency_map)).astype(np.uint8)
    else:
        saliency_map = (saliency_map * 255).astype(np.uint8)
    
    return saliency_map

def segment_image_regions(image, saliency_map, num_regions=16):
    """
    Divide image into regions based on grid and compute regional properties
    """
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
            
            # Calculate region weight based on saliency
            weight = np.mean(saliency_region) / 255.0
            
            regions.append(region)
            region_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(region_weights)
    region_weights = [w / total_weight for w in region_weights]
    
    return regions, region_weights

def compute_regional_phash(region):
    """Compute perceptual hash for a region"""
    region_pil = Image.fromarray(cv.cvtColor(region, cv.COLOR_BGR2RGB))
    phash = imagehash.phash(region_pil, hash_size=8)
    return phash

# ------------------ Step 4 & 5: Adaptive Weighting and Merging ------------------
def compute_weighted_simhash(image, num_regions=16):
    """
    Compute weighted SimHash from regional perceptual hashes
    Steps 2-5 of the algorithm
    """
    # Compute saliency map
    saliency_map = compute_saliency_map(image)
    
    # Segment image into regions
    regions, weights = segment_image_regions(image, saliency_map, num_regions)
    
    # Compute perceptual hash for each region
    regional_hashes = []
    for region in regions:
        phash = compute_regional_phash(region)
        regional_hashes.append(phash)
    
    # Merge weighted hashes into final SimHash
    # Convert hashes to binary arrays and apply weights
    hash_size = 64  # 8x8 phash
    final_hash = np.zeros(hash_size)
    
    for phash, weight in zip(regional_hashes, weights):
        hash_array = np.array([int(b) for b in bin(int(str(phash), 16))[2:].zfill(hash_size)])
        final_hash += hash_array * weight
    
    # Threshold to create binary hash
    final_hash = (final_hash > 0.5).astype(int)
    
    # Convert to hex string for storage
    hash_value = int(''.join(map(str, final_hash)), 2)
    
    return hash_value, regional_hashes, weights

# ------------------ Step 6: Hamming Distance Calculation ------------------
def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two hash values"""
    return bin(hash1 ^ hash2).count('1')

# ------------------ Step 7: Image Retrieval and Matching ------------------
def detect_forgery(image_path, dataset_folder, hamming_threshold=15, top_n=5):
    """
    Complete forgery detection pipeline following the algorithm
    """
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Error: Dataset folder not found {dataset_folder}")
    
    print(f"\n{'='*60}")
    print(f"Analyzing image: {os.path.basename(image_path)}")
    print(f"{'='*60}\n")
    
    # Step 1: Preprocess input image
    print("Step 1: Preprocessing input image...")
    query_image_rgb, query_image_bgr = preprocess_image(image_path)
    
    # Steps 2-5: Extract features and compute weighted SimHash
    print("Steps 2-5: Extracting features and computing weighted SimHash...")
    query_hash, _, _ = compute_weighted_simhash(query_image_bgr)
    query_deep_features = extract_deep_features(query_image_rgb)
    
    # Step 7: Compare with dataset
    print(f"Step 7: Comparing with dataset images...\n")
    
    matches = []
    
    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                # Preprocess database image
                db_image_rgb, db_image_bgr = preprocess_image(file_path)
                
                # Compute hash and features
                db_hash, _, _ = compute_weighted_simhash(db_image_bgr)
                db_deep_features = extract_deep_features(db_image_rgb)
                
                # Step 6: Calculate Hamming distance
                ham_dist = hamming_distance(query_hash, db_hash)
                
                # Additional: Cosine similarity for deep features
                cos_sim = cosine_similarity([query_deep_features], [db_deep_features])[0][0]
                
                matches.append({
                    'filename': file,
                    'hamming_distance': ham_dist,
                    'cosine_similarity': cos_sim,
                    'combined_score': ham_dist - (cos_sim * 20)  # Lower is better
                })
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
    
    # Sort by combined score (lower is better)
    matches.sort(key=lambda x: x['combined_score'])
    
    # Display results
    print(f"\n{'='*60}")
    print(f"TOP {min(top_n, len(matches))} MATCHES:")
    print(f"{'='*60}\n")
    
    for i, match in enumerate(matches[:top_n], 1):
        print(f"Rank {i}: {match['filename']}")
        print(f"  Hamming Distance: {match['hamming_distance']}")
        print(f"  Cosine Similarity: {match['cosine_similarity']:.4f}")
        print(f"  Combined Score: {match['combined_score']:.2f}")
        
        if match['hamming_distance'] <= hamming_threshold:
            print(f"  ⚠️  POTENTIAL FORGERY DETECTED (Below threshold)")
        print()
    
    # Return best match
    if matches:
        best_match = matches[0]
        if best_match['hamming_distance'] <= hamming_threshold:
            print(f"✓ CONCLUSION: Image is similar to '{best_match['filename']}'")
            print(f"  This suggests potential forgery or manipulation.")
        else:
            print(f"✓ CONCLUSION: No significant similarity found.")
            print(f"  Image appears to be authentic or heavily modified.")
    else:
        print("No matches found in dataset.")
    
    return matches

# ------------------ Dataset Merging Utility ------------------
def merge_datasets(kaggle_root, my_authentic, my_fraud, merged_root):
    """Merge Kaggle dataset with custom dataset"""
    dataset_folder = os.path.join(kaggle_root, "Dataset")
    kaggle_original = os.path.join(dataset_folder, "Original")
    kaggle_forged = os.path.join(dataset_folder, "Forged")
        
    merged_original = os.path.join(merged_root, "Original")
    merged_forged = os.path.join(merged_root, "Forged")

    # Clean up old merged dataset if it exists
    if os.path.exists(merged_root):
        shutil.rmtree(merged_root)

    os.makedirs(merged_original, exist_ok=True)
    os.makedirs(merged_forged, exist_ok=True)

    # Copy Kaggle Originals
    if os.path.exists(kaggle_original):
        for f in os.listdir(kaggle_original):
            src = os.path.join(kaggle_original, f)
            dst = os.path.join(merged_original, f"Kaggle_{f}")
            shutil.copy(src, dst)

    # Copy Kaggle Forged
    if os.path.exists(kaggle_forged):
        for f in os.listdir(kaggle_forged):
            src = os.path.join(kaggle_forged, f)
            dst = os.path.join(merged_forged, f"Kaggle_{f}")
            shutil.copy(src, dst)

    # Copy your Authentic
    if os.path.exists(my_authentic):
        for f in os.listdir(my_authentic):
            src = os.path.join(my_authentic, f)
            dst = os.path.join(merged_original, f"MyAuth_{f}")
            shutil.copy(src, dst)

    # Copy your Fraud
    if os.path.exists(my_fraud):
        for f in os.listdir(my_fraud):
            src = os.path.join(my_fraud, f)
            dst = os.path.join(merged_forged, f"MyFraud_{f}")
            shutil.copy(src, dst)

    print(f"Dataset merged successfully to: {merged_root}")
    return merged_root

# ------------------ Main Execution ------------------
def main():
    # Download Kaggle dataset
    print("Downloading Kaggle dataset...")
    dataset_root = kagglehub.dataset_download("labid93/image-forgery-detection")
    print(f"Dataset downloaded to: {dataset_root}")

    my_authentic = "DataSet/Authentic"
    my_fraud = "DataSet/Fraud"
    merged_root = "MergedDataSet"
    
    # Merge datasets
    print("\nMerging datasets...")
    dataset_path = merge_datasets(dataset_root, my_authentic, my_fraud, merged_root)

    # Test forgery detection
    print("\n" + "="*60)
    print("STARTING FORGERY DETECTION")
    print("="*60)
    
    # Example: Test with a fraud image
    test_image = os.path.join(my_fraud, "12857.jpg")
    dataset_folder = os.path.join(dataset_path, "Original")

    try:
        matches = detect_forgery(test_image, dataset_folder, hamming_threshold=15, top_n=5)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

if __name__ == "__main__":
    main()