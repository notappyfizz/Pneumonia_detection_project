import os
import torch
from model import CNNModel
from train import train_model
from dataset import load_data

# Path to the full training dataset
TRAIN_DATA_PATH = "/Users/aparajitasrinivasan/Downloads/chest_xray/train"  

# Make sure checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# Load training data
train_loader = load_data(TRAIN_DATA_PATH)

# Initialize model
model = CNNModel()

# Train the model
train_model(model, train_loader)

# Save trained model weights
torch.save(model.state_dict(), "checkpoints/final_weights.pth")
print("✅ Model trained and saved as checkpoints/final_weights.pth")
