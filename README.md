# Pneumonia_detection_project
# Pneumonia Detection using Chest X-Ray Images

## Project Description
This project builds a deep learning model to detect pneumonia from grayscale chest X-ray images. The task is a binary classification: pneumonia-positive vs normal.  
The model was built **from scratch** using **PyTorch** without any pretrained networks, to deeply understand CNN architectures.

## Choice of Model
A custom Convolutional Neural Network (CNN) was implemented with:
- 3 convolutional layers
- Max pooling layers for downsampling
- Batch normalization and dropout layers to prevent overfitting
- Fully connected layers for final classification

The model is trained to output a Boolean:
- `True` → Pneumonia detected
- `False` → Normal chest X-ray

## Dataset Information
The dataset used is [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.  
It contains labeled X-ray images of pediatric patients categorized into "NORMAL" and "PNEUMONIA".

### How to Download the Dataset
1. Go to [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. Download and unzip it.
3. Place the full `train/` folder in your working directory.

## Installation Instructions
You need:
- Python 3.8+
- PyTorch
- torchvision
- pillow

You can install them using:
```bash
pip install torch torchvision pillow
