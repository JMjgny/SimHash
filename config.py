# Configuration constants
IMAGE_SIZE = (224, 224)
NUM_REGIONS = 16
HASH_SIZE = 64
HAMMING_THRESHOLD = 15
TOP_N_MATCHES = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

USE_FULL_PREPROCESSING = False  # Set to True for high quality, False for speed
USE_CLAHE = True
USE_DENOISING = False  # This is the slowest operation
USE_SHARPENING = False

# GPU Configuration
USE_GPU = True
GPU_ID = 0  # Use first GPU (change if you have multiple GPUs)
BATCH_SIZE = 32  # For batch processing multiple images
