import torch

import parameters


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


# Gets appropriate device
def get_device():
    if parameters.force_device != "":
        device = torch.device(parameters.force_device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\nUsing device: {device}\n")
    return device


# Validates the model
def validate(model, validate_loader, device):
    print("Validation:")
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validate_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # print the validation results
            print("--------------------------------------------------------")
            print(f"[{batch_idx + 1}] predicted: {output.tolist()[0]}")
            print(f"[{batch_idx + 1}] actual: {target.tolist()[0]}")
            print(f"\033[1;34m[{batch_idx + 1}] difference: {(target - output).pow(2).sqrt().tolist()[0]}\033[0m")
            print("--------------------------------------------------------\n")
