# interface.py

# replace MyCustomModel with the name of your model
from model import CNNModel as TheModel

# change my_descriptively_named_train_function to the function inside train.py that runs the training loop
from train import train_model as the_trainer

# change cryptic_inf_f to the function inside predict.py that can generate inference on a single image/batch
from predict import predict_images as the_predictor

# change UnicornImgDataset to your custom Dataset class
from dataset import ChestXRayDataset as TheDataset

# change unicornLoader to your custom dataloader
from dataset import load_data as the_dataloader

# hyperparameters from config
from config import batch_size as the_batch_size
from config import epochs as total_epochs
