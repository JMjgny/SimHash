import os

dataset_root = r"C:\Users\Marc\.cache\kagglehub\datasets\labid93\image-forgery-detection\versions\1"

print("Dataset root:", dataset_root)
print("\nContents of dataset folder:\n")

for root, dirs, files in os.walk(dataset_root):
    print("📂 Folder:", root)
    for d in dirs:
        print("   Subdir:", d)
    for f in files[:10]:  # only show first 10 files
        print("   File:", f)
    break  # only look at top-level once