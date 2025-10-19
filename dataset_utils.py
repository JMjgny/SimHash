import os
import shutil

def merge_datasets(kaggle_root, my_authentic, my_fraud, merged_root):
    """Merge Kaggle dataset with custom dataset"""
    dataset_folder = os.path.join(kaggle_root, "Dataset")
    kaggle_original = os.path.join(dataset_folder, "Original")
    kaggle_forged = os.path.join(dataset_folder, "Forged")
        
    merged_original = os.path.join(merged_root, "Original")
    merged_forged = os.path.join(merged_root, "Forged")

    if os.path.exists(merged_root):
        shutil.rmtree(merged_root)

    os.makedirs(merged_original, exist_ok=True)
    os.makedirs(merged_forged, exist_ok=True)

    if os.path.exists(kaggle_original):
        for f in os.listdir(kaggle_original):
            src = os.path.join(kaggle_original, f)
            dst = os.path.join(merged_original, f"Kaggle_{f}")
            shutil.copy(src, dst)

    if os.path.exists(kaggle_forged):
        for f in os.listdir(kaggle_forged):
            src = os.path.join(kaggle_forged, f)
            dst = os.path.join(merged_forged, f"Kaggle_{f}")
            shutil.copy(src, dst)

    if os.path.exists(my_authentic):
        for f in os.listdir(my_authentic):
            src = os.path.join(my_authentic, f)
            dst = os.path.join(merged_original, f"MyAuth_{f}")
            shutil.copy(src, dst)

    if os.path.exists(my_fraud):
        for f in os.listdir(my_fraud):
            src = os.path.join(my_fraud, f)
            dst = os.path.join(merged_forged, f"MyFraud_{f}")
            shutil.copy(src, dst)

    print(f"Dataset merged successfully to: {merged_root}")
    return merged_root
