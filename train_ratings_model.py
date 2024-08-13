import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import csv
from sklearn.model_selection import KFold
import argparse

# Define the ResNet-based model
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)

# Example dataset and dataloader setup
class PortfolioDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['filepath']
        rating = self.data.iloc[idx]['rating']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(rating, dtype=torch.float32)

# Load the CSV with ratings
ratings_df = pd.read_csv('ratings_adjusted.csv')
screenshot_dir = "screenshots"
ratings_df['filepath'] = ratings_df['url'].apply(lambda x: os.path.join(screenshot_dir, x))
ratings_df = ratings_df[ratings_df['filepath'].apply(os.path.exists)]

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Argument parser for loading a model checkpoint
parser = argparse.ArgumentParser(description="Train a ResNet model with optional checkpoint loading.")
parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint to load.')
args = parser.parse_args()

# Initialize the model
model = ResNetModel()

# Load model checkpoint if provided
if args.checkpoint:
    print(f"Loading model checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed training from epoch {start_epoch}.")
else:
    start_epoch = 0

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# K-Fold Cross Validation Setup
def k_fold_cross_validation(df, k=5, num_epochs=50, save_interval=1, checkpoint_dir='checkpoints', loss_csv='loss_log.csv'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare the CSV file to log loss values
    with open(loss_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold', 'Epoch', 'Training Loss', 'Validation Loss'])

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print(f"\nStarting Fold {fold + 1}/{k}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0

            with tqdm(total=len(train_loader), desc=f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for images, ratings in train_loader:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), ratings)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix({'Training Loss': running_loss / (pbar.n + 1)})
                    pbar.update()

            train_loss = running_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, ratings in val_loader:
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), ratings)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")

            # Save the model checkpoint every `save_interval` epochs
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold + 1}_model_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")

            # Log the losses to the CSV file
            with open(loss_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fold + 1, epoch + 1, train_loss, val_loss])

if __name__ == "__main__":
    dataset = PortfolioDataset(ratings_df, transform=transform)
    k_fold_cross_validation(ratings_df, k=5, num_epochs=50)