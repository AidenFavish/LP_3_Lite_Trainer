import torch
from PIL import Image


# Function to save the model state dict
def save_model_state(model, path='no_name_model.pth'):
    torch.save(model.state_dict(), path)


# Function to load the model state dict
def load_model_state(model, path):
    model.load_state_dict(torch.load(path))
    return model


# Function to save the entire model
def save_model(model, path):
    torch.save(model, path)


# Function to load the entire model
def load_model(path):
    model = torch.load(path)
    return model


# Function to test the model with an input image (will verify more later)
def test_model(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move the image to the same device as the model
    with torch.no_grad():
        output = model(image)
    return output


# Gets appropriate device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("Using device: ", device)
    return device
