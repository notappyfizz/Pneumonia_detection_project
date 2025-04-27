import torch
from PIL import Image
import torchvision.transforms as transforms
from config import resize_x, resize_y

def predict_images(model, img_paths):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor()
    ])
    predictions = []
    for path in img_paths:
        img = Image.open(path).convert('L')
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            pred = output.item() > 0.5
            predictions.append(pred)
    return predictions
