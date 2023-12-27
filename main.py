import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir=runs
import time
import torch.nn.functional as F
import model_architectures
import dataset_loader
import parameters
import training_tools
import trainer


# Main function
def main():
    # DEFAULT MODEL
    default_model = model_architectures.DefaultCNN()

    # Ask user to train a new model or load an existing one
    load_input = input(
        "Do you want to use a blank model or load a model? (blank/load full/load state): ").strip().lower()
    if load_input == 'load full':
        model_path = input("Enter the path of the model to load full: ").strip()
        model = training_tools.load_model(model_path)
    elif load_input == 'load state':
        model_path = input("Enter the path of the model to load state: ").strip()
        model = training_tools.load_model_state(default_model, model_path)
    else:
        model = default_model

    # Prepare the device
    device = training_tools.get_device()
    model.to(device)  # Move the model to the MPS device or CPU

    # Load Data
    dataset = dataset_loader.CustomDataset(csv_file=parameters.training_folder + '/data.csv',
                                           root_dir=parameters.training_folder, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=dataset, batch_size=parameters.batch_size, shuffle=True)

    # Train the model
    trainer.train(model, train_loader, device)

    # Offer to save the model after training
    save_input = input("Do you want to save the model? How? (full/state/no): ").strip().lower()
    if save_input == 'full':
        save_model_path = input("Enter the path to save the model in full: ")
        training_tools.save_model(model, save_model_path)
    elif save_input == 'state':
        save_model_path = input("Enter the path to save the model state: ")
        training_tools.save_model_state(model, save_model_path)

    # Offer to run validation
    test_input = input("Do you want to test the model? (yes/no): ").strip().lower()
    if test_input == 'yes':
        test_image_path = input("Enter the path of the image to test: ")
        transform = transforms.Compose([transforms.ToTensor()])
        output = test_model(model, test_image_path, transform, device)
        print("Predicted outputs:", output)


if __name__ == "__main__":
    main()
