import os
import pandas as pd

from SOP1 import run_sop1_result
from Sopv2 import run_sop2_result
from SOP3 import run_sop3_result
from SOP4 import run_sop4_result
from forgery_detector import run_improved_pairwise 
from feature_extractor import get_feature_extractor  # function to initialize extractor

# Dataset root and image paths (ensure correct casing and spelling)
dataset_root = r"C:\Users\Marc\.cache\kagglehub\datasets\labid93\image-forgery-detection\versions\1"

original_img = os.path.join(dataset_root, "DataSet", "Original", "654.jpg")
forged_imgs = {
    "Lighting Increased": os.path.join(dataset_root, "DataSet", "Forged", "bright_new.jpg"),
    "Noise": os.path.join(dataset_root, "DataSet", "Forged", "11039.jpg"),
    "Blurred": os.path.join(dataset_root, "DataSet", "Forged", "blur_new.jpg"),
    "Face Blurred": os.path.join(dataset_root, "DataSet", "Forged", "11037.jpg"),
}

def compare_single_pair(original, modified, feature_extractor):
    # Run each SOP method comparing the pair
    sop1_result = run_sop1_result(original, modified)
    sop2_result = run_sop2_result(original, modified)
    sop3_result = run_sop3_result(original, modified)
    sop4_result = run_sop4_result(original, modified)
    improved_result = run_improved_pairwise(original, modified)

    return {
        "SOP1": sop1_result,
        "SOP2": sop2_result,
        "SOP3": sop3_result,
        "SOP4": sop4_result,
        "Improved": improved_result
    }

def main():
    feature_extractor = get_feature_extractor()
    results = []

    for label, forged_img in forged_imgs.items():
        print(f"\nComparing pair: Original vs {label}")
        result = compare_single_pair(original_img, forged_img, feature_extractor)
        results.append({"Case": label, **result})

        # Print results in readable format
        for key, val in result.items():
            print(f"{key} result:\n{val}\n")

    # Save summary CSV
    df = pd.DataFrame(results)
    df.to_csv("pairwise_comparison_results.csv", index=False)
    print("Results saved to pairwise_comparison_results.csv")

if __name__ == "__main__":
    main()
