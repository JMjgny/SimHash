import torch

print("="*60)
print("PYTORCH INSTALLATION INFO")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
print("="*60)
