import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import resize_x, resize_y, input_channels, batch_size

class ChestXRayDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((resize_x, resize_y)),
            transforms.ToTensor()
        ])
        
        # Walk through all subdirectories (e.g., NORMAL/, PNEUMONIA/)
        for subdir, _, files in os.walk(data_dir):
            for filename in files:
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                filepath = os.path.join(subdir, filename)
                label = 1 if 'pneumonia' in subdir.lower() else 0
                self.data.append(filepath)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('L')
        img = self.transform(img)
        return img, self.labels[idx]

def load_data(data_dir):
    dataset = ChestXRayDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
