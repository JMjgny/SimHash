import os
import torch
import kagglehub
from dataset_utils import merge_datasets
from forgery_detector import detect_forgery
from feature_extractor import get_feature_extractor

def main():
    # Check CUDA availability
    print("="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        print(f"✓ Current GPU: {torch.cuda.current_device()}")
    else:
        print("✗ CUDA is NOT available. Using CPU.")
        print("  Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("="*60 + "\n")
    
    print("Downloading Kaggle dataset...")
    dataset_root = kagglehub.dataset_download("labid93/image-forgery-detection")
    print(f"Dataset downloaded to: {dataset_root}")

    my_authentic = "DataSet/Authentic"
    my_fraud = "DataSet/Fraud"
    merged_root = "MergedDataSet"
    
    print("\nMerging datasets...")
    dataset_path = merge_datasets(dataset_root, my_authentic, my_fraud, merged_root)

    print("\n" + "="*60)
    print("STARTING FORGERY DETECTION")
    print("="*60)
    
    # Initialize feature extractor (automatically uses GPU if available)
    feature_extractor = get_feature_extractor()
    
    test_image = os.path.join(my_fraud, "11492.jpg")
    dataset_folder = os.path.join(dataset_path, "Original")

    try:
        matches = detect_forgery(test_image, dataset_folder, feature_extractor)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n✓ GPU memory cleared")

if __name__ == "__main__":
    main()
