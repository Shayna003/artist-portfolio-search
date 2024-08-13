import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pymongo import MongoClient
from tqdm import tqdm
import argparse

# Define the ResNet-based model
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)

# Dataset class for loading images from MongoDB
class PortfolioDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]['screenshot_path']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.data[idx]['url']

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the model
model = ResNetModel()

# Load the trained model checkpoint
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode
print("Model loaded and ready for inference.")

# MongoDB connection
client = MongoClient('localhost', 27017)
db = client['portfolio_db']
collection = db['portfolios']

# Fetch all data from the database
data = list(collection.find({}))

# Create a DataLoader
dataset = PortfolioDataset(data, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Iterate over the dataset and predict ratings
with torch.no_grad():
    for images, urls in tqdm(dataloader, desc="Rating portfolios"):
        outputs = model(images).squeeze().numpy()  # Get predictions

        # Update the ratings in the database
        for url, rating in zip(urls, outputs):
            collection.update_one({'url': url}, {'$set': {'rating': float(rating)}})

print("All entries have been rated and updated in the database.")
