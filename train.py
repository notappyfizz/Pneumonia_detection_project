import torch
import torch.nn as nn
import torch.optim as optim
from config import epochs, learning_rate

def train_model(model, train_loader):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in train_loader:
            labels = labels.float().unsqueeze(1)  # [batch_size, 1]
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
