import cv2 as cv
import numpy as np
from simhash import Simhash
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import os
import pywt
from torch.utils.data import DataLoader

# Define a Simple CNN Model for Feature Extraction
class ForgeryCNN(nn.Module):
    def __init__(self):
        super(ForgeryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and Train the CNN Model
def train_cnn(dataset_path, model_save_path, epochs=10, batch_size=32, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ForgeryCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), model_save_path)
    print("Model training complete. Saved to", model_save_path)

# Load pre-trained model
cnn_model = ForgeryCNN()
model_path = "cnn_model.pth"
if os.path.exists(model_path):
    cnn_model.load_state_dict(torch.load(model_path))
cnn_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def preprocess_image(image_path, size=(64, 64)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image not found {image_path}")
    
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read image {image_path}")
    
    image = cv.GaussianBlur(image, (3, 3), 0)
    image = cv.resize(image, size)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  
    
    return image

def image_to_feature_vector(image_path):
    image = preprocess_image(image_path)
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray_image, 100, 200)
    
    coeffs2 = pywt.dwt2(gray_image, 'haar')
    cA, (cH, cV, cD) = coeffs2
    wavelet_features = np.array([np.mean(np.abs(cH)), np.mean(np.abs(cV)), np.mean(np.abs(cD))])
    
    orb = cv.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    orb_features = np.mean(descriptors, axis=0) if descriptors is not None else np.zeros(32)
    
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        deep_features = cnn_model(image_tensor).flatten().numpy()
    
    feature_vector = np.concatenate((wavelet_features, orb_features, deep_features))
    feature_vector = (feature_vector - np.min(feature_vector)) / (np.max(feature_vector) - np.min(feature_vector) + 1e-5)
    
    return feature_vector

def compute_simhash(image_path):
    feature_vector = image_to_feature_vector(image_path)
    feature_groups = [str(int(x * 1000)) for x in feature_vector[:128]]  
    return Simhash(' '.join(feature_groups), f=128)

def hamming_distance(hash1, hash2):
    return hash1.distance(hash2)

def detect_forgery(image_path, dataset_folder, threshold=40):
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Error: Dataset folder not found {dataset_folder}")
    
    image_hash = compute_simhash(image_path)
    print(f"Scanning dataset folder: {dataset_folder}")
    print("Files in dataset folder:", os.listdir(dataset_folder))
    
    closest_match = None
    min_distance = float('inf')
    
    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        if os.path.isfile(file_path):
            print(f"Processing: {file}")
            dataset_hash = compute_simhash(file_path)
            distance = hamming_distance(image_hash, dataset_hash)
            print(f"{file}: Hamming Distance = {distance}")
            
            if distance < min_distance:
                min_distance = distance
                closest_match = file
    
    if closest_match and min_distance <= threshold:
        print(f"Selected image is similar to: {closest_match} (Hamming Distance: {min_distance})")
    else:
        print("No significant similarity found.")

def main():
    dataset_path = "DataSet"  # Path to training dataset
    model_save_path = "cnn_model.pth"
    train_cnn(dataset_path, model_save_path)  # Train the CNN
    
    image1 = "DataSet/Fraud/image2.jpg"
    dataset_folder = "DataSet/Authentic"
    
    try:
        detect_forgery(image1, dataset_folder)
    except FileNotFoundError as e:
        print(e)
        return

if __name__ == "__main__":
    main()
