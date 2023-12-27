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


# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate outputs
        # output = (input - kernel_size + 2 * padding) / stride + 1
        self.conv_output_size = (1000 - 3 + 2 * 1) // 1 + 1
        # Calculate pooling output size
        self.pool_output_size = self.conv_output_size // 2

        self.fc1 = nn.Linear(16 * self.pool_output_size * self.pool_output_size, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = x.view(-1, 16 * self.pool_output_size * self.pool_output_size)
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN2(nn.Module):
    def __init__(self):
        super(SimpleCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the output size after the first conv and pool
        self.conv1_output_size = (1000 - 3 + 2 * 1) // 2 + 1  # After conv1
        self.pool1_output_size = self.conv1_output_size // 2  # After pool1

        # Calculate the output size after the second conv and pool
        self.conv2_output_size = (self.pool1_output_size - 3 + 2 * 1) // 2 + 1  # After conv2
        self.pool2_output_size = self.conv2_output_size // 2  # After pool2

        # Fully connected layers
        self.fc1 = nn.Linear(32 * self.pool2_output_size * self.pool2_output_size, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act1(self.conv2(x)))
        x = x.view(-1, 32 * self.pool2_output_size * self.pool2_output_size)
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNNV3(nn.Module):
    def __init__(self):
        super(SimpleCNNV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the output size after the first conv and pool
        self.conv1_output_size = (1000 - 3 + 2 * 1) // 2 + 1  # After conv1
        self.pool1_output_size = self.conv1_output_size // 2  # After pool1

        # Calculate the output size after the second conv and pool
        self.conv2_output_size = (self.pool1_output_size - 3 + 2 * 1) // 2 + 1  # After conv2
        self.pool2_output_size = self.conv2_output_size // 2  # After pool2

        # Flattened size of the grayscale input image
        self.input_flattened_size = 1000 * 1000

        # Fully connected layers
        # The first fully connected layer takes the concatenated output of the last pooling layer
        # and the flattened grayscale input image
        self.fc1_input_size = 32 * self.pool2_output_size * self.pool2_output_size + self.input_flattened_size
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Save the original grayscale image
        grayscale_x = torch.mean(x, dim=1, keepdim=True)  # Convert to grayscale
        grayscale_flattened = torch.flatten(grayscale_x, 1)  # Flatten the grayscale image

        # Process through convolutional and pooling layers
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act1(self.conv2(x)))
        #conv_flattened = torch.flatten(x, 1)  # Flatten the output of the last pooling layer
        conv_flattened = x.view(-1, 32 * self.pool2_output_size * self.pool2_output_size)

        # Concatenate the outputs of the last pooling layer with the flattened grayscale image
        x = torch.cat((conv_flattened, grayscale_flattened), dim=1)

        # Process through fully connected layers
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x


# Dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        frame_number = self.annotations.iloc[index, 0]
        img_name = f'frame{frame_number}.0.jpg'  # Construct image filename
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        y_label = torch.tensor(
            self.annotations.iloc[index, 1:5].values.astype('float32'))  # centerX, centerY, radius, turn

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Function to save the model
def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)


# Function to load the model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


# Function to test the model with an input image
def test_model(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move the image to the same device as the model
    with torch.no_grad():
        output = model(image)
    return output


def train(num_epochs, dataloader, model, criterion, optimizer, device):
    print('Starting Training\n')
    # TensorBoard setup
    writer = SummaryWriter(f"runs/simple_cnn/{time.time()}")
    for epoch in range(num_epochs):
        print("------------------------------------------")
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            avg_loss = running_loss / (i + 1)  # Average loss for the epoch so far
            writer.add_scalar('training loss', avg_loss, epoch * len(dataloader) + i)
            if i % 10 == 0:
                print(f'[{epoch + 1}, {i + 1}] avg loss: {avg_loss:.3f}')
        print(f"------------------------------------------\n\033[1;31mEpoch {epoch + 1} finished. Average Loss: {running_loss/len(dataloader)}\033[0m\n")


# Main
def main():
    # Ask user to train a new model or load an existing one
    user_input = input("Do you want to use a blank model or load a model? (blank/load): ").strip().lower()
    model = SimpleCNN()
    if user_input == 'load':
        model_path = input("Enter the path of the model to load: ").strip()
        model = load_model(model, model_path)

    # Set device to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)  # Move the model to the MPS device or CPU

    # Hyperparameters
    learning_rate = 0.002
    weight_decay = 1e-5
    batch_size = 15
    num_epochs = 1

    # Load Data
    dataset = CustomDataset(csv_file='/Users/aiden/Desktop/Training2/data.csv',
                            root_dir='/Users/aiden/Desktop/Training2', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model
    train(num_epochs, train_loader, model, criterion, optimizer, device)

    # Save the model after training
    save_model_path = input("Enter the path to save the model: ")
    save_model(model, save_model_path)

    # Ask if the user wants to test the model
    test_input = input("Do you want to test the model? (yes/no): ").strip().lower()
    if test_input == 'yes':
        test_image_path = input("Enter the path of the image to test: ")
        transform = transforms.Compose([transforms.ToTensor()])
        output = test_model(model, test_image_path, transform, device)
        print("Predicted outputs:", output)


if __name__ == "__main__":
    main()
