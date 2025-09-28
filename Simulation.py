
import os
import kagglehub
import cv2 as cv
import numpy as np
import pywt
from simhash import Simhash
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split

# ------------------ CNN Model ------------------
class ForgeryCNN(nn.Module):
    def __init__(self):
        super(ForgeryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------ Data Loading with Augmentation ------------------
def load_data(dataset_path, batch_size=16):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Compute class weights to balance fraud and authentic
    class_counts = np.bincount(dataset.targets)
    class_weights = 1. / class_counts
    weights = class_weights[dataset.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# ------------------ Training the CNN ------------------
def train_cnn(dataset_path, model_save_path, epochs=10, batch_size=16, learning_rate=0.001):
    train_loader, val_loader = load_data(dataset_path, batch_size)
    
    model = ForgeryCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), model_save_path)
    print("Model training complete. Saved to", model_save_path)

# ------------------ Feature Extraction ------------------
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
    
    feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-5)
    
    return feature_vector

def compute_simhash(feature_vector):
    feature_groups = [str(int(x * 1000)) for x in feature_vector[:128]]
    return Simhash(' '.join(feature_groups), f=128)

def hamming_distance(hash1, hash2):
    return hash1.distance(hash2)

# ------------------ Forgery Detection ------------------
def detect_forgery(image_path, dataset_folder, hamming_threshold=40, cosine_threshold=0.85):
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Error: Dataset folder not found {dataset_folder}")
    
    print(f"Scanning dataset folder: {dataset_folder}")
    print("Files in dataset folder:", os.listdir(dataset_folder))

    query_vector = image_to_feature_vector(image_path)
    query_hash = compute_simhash(query_vector)
    
    closest_match = None
    min_combined_score = float('inf')
    best_details = None

    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        if os.path.isfile(file_path):
            print(f"Processing: {file}")
            db_vector = image_to_feature_vector(file_path)
            db_hash = compute_simhash(db_vector)
            
            ham_dist = hamming_distance(query_hash, db_hash)
            cos_sim = cosine_similarity([query_vector], [db_vector])[0][0]

            print(f"{file}: Hamming Distance = {ham_dist}, Cosine Similarity = {cos_sim:.4f}")
            
            combined_score = ham_dist - (cos_sim * 100)

            if combined_score < min_combined_score:
                min_combined_score = combined_score
                closest_match = file
                best_details = (ham_dist, cos_sim)
    
    if closest_match and best_details:
        ham_dist, cos_sim = best_details
        if ham_dist <= hamming_threshold and cos_sim >= cosine_threshold:
            print(f"Selected image is similar to: {closest_match} (Hamming Distance: {ham_dist}, Cosine Similarity: {cos_sim:.4f})")
        else:
            print("No significant similarity found.")
    else:
        print("No images processed.")

# ------------------ Main Function ------------------
def main():
    dataset_path = kagglehub.dataset_download("labid93/image-forgery-detection")
    model_save_path = "cnn_model.pth"
    train_cnn(dataset_path, model_save_path)  # Train the CNN

    image1 = "DataSet/Fraud/6(2).jpg"
    dataset_folder = "DataSet/Authentic"

    try:
        detect_forgery(image1, dataset_folder)
    except FileNotFoundError as e:
        print(e)
        return

if __name__ == "__main__":
    main()
