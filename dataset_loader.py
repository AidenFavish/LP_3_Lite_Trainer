import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as transforms


# Loads the dataset from the given directories and returns a DataLoader object.
class LPDataset1(Dataset):
    def __init__(self, project_dir, transform=None):
        self.annotations = pd.read_csv(project_dir + "/data.csv")
        self.root_dir = project_dir
        if transform is None:
            # Define default transform if none provided
            self.transform = transforms.Compose([
                transforms.Resize((1000, 1000)),  # Resize the image
                transforms.ToTensor()  # Convert the image to a tensor
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        frame_number = self.annotations.iloc[index, 0]  # Get the frame number
        img_name = f'frame{frame_number}.0.jpg'  # Construct image filename
        img_path = os.path.join(self.root_dir, img_name)  # Construct image path
        image = Image.open(img_path).convert('RGB')  # Convert to RGB

        # get labels in tensor with centerX, centerY, radius, turn
        y_label = torch.tensor(self.annotations.iloc[index, 1:5].values.astype('float32'))

        # Apply the transform to the image
        image = self.transform(image)

        return image, y_label


# Loads the dataset from the given directories and returns a DataLoader object.
class LPDataset2(Dataset):
    def __init__(self, project_dir, transform=None):
        self.annotations = pd.read_csv(project_dir + "/data.csv")
        self.root_dir = project_dir
        if transform is None:
            # Define default transform if none provided
            self.transform = transforms.Compose([
                transforms.Resize((1000, 1000)),  # Resize the image
                transforms.ToTensor()  # Convert the image to a tensor
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        frame_number = self.annotations.iloc[index, 0]  # Get the frame number
        img_name = f'frame{frame_number}.0.jpg'  # Construct image filename
        img_path = os.path.join(self.root_dir, img_name)  # Construct image path
        image = Image.open(img_path).convert('RGB')  # Convert to RGB

        # get labels in tensor with centerX, centerY, radius, turn
        y_label = torch.tensor(self.annotations.iloc[index, 1:3].values.astype('float32'))

        # Apply the transform to the image
        image = self.transform(image)

        return image, y_label
