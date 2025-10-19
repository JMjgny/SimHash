import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_feature_extractor():
    """Initialize and return ResNet-50 feature extractor on GPU"""
    extractor = ResNet50FeatureExtractor()
    extractor = extractor.to(device)  # Move model to GPU
    extractor.eval()
    return extractor

def extract_deep_features(image_rgb, extractor):
    """Extract high-level features using ResNet-50 with GPU acceleration"""
    image_tensor = transform(image_rgb).unsqueeze(0)
    image_tensor = image_tensor.to(device)  # Move tensor to GPU
    
    with torch.no_grad():
        features = extractor(image_tensor)
    
    # Move result back to CPU for NumPy conversion
    return features.cpu().flatten().numpy()
