from torch.utils.data import DataLoader
from torchvision import transforms
import model_architectures
import dataset_loader
import parameters
import model_tools
import trainer
import torch
import torch.nn as nn


# Main function
def main():
    # DEFAULT MODEL
    default_model = model_architectures.ImprovedCNN()

    # Ask user to train a new model or load an existing one
    load_input = input(
        "Do you want to use a blank model or load a model? (blank/state/full): ").strip().lower()
    if load_input == 'full':
        model = model_tools.load_model(parameters.model_path)
    elif load_input == 'state':
        model = model_tools.load_model_state(default_model, parameters.model_path)
    else:
        model = default_model

    # Prepare the device
    device = model_tools.get_device()
    model.to(device)  # Move the model to the MPS device or CPU

    # Check for parallel processing
    if parameters.parallel_processing and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Parallel processing enabled.\n")
    elif parameters.parallel_processing:
        print("Parallel processing not available.\n")
        parameters.parallel_processing = False

    # Skips training if the user wants to only validate otherwise trains and saves model here
    if not parameters.only_validate:
        # Load training data
        training_dataset = dataset_loader.LPDataset1(project_dir=parameters.training_folder,
                                                     transform=transforms.ToTensor())
        train_loader = DataLoader(dataset=training_dataset, batch_size=parameters.batch_size, shuffle=True,
                                  num_workers=2)

        # Train the model
        # trainer.train(model, train_loader, device)  # Standard training
        trainer.mp_train()  # Multi-processing training

        # Save the non-parallel model
        if parameters.parallel_processing:
            model = model.module

        # Offer to save the model after training or automatically save it
        if parameters.automate_save != "":
            save_input = parameters.automate_save
        else:
            save_input = input("Do you want to save the model? How? (full/state/no): ").strip().lower()

        if save_input == 'full':
            model_tools.save_model(model, parameters.save_as)
        elif save_input == 'state':
            model_tools.save_model_state(model, parameters.save_as)

    # Offer to run validation
    validate_user_input = input("Do you want to validate the model? (yes/no): ").strip().lower()
    if validate_user_input == 'yes':
        validation_dataset = dataset_loader.LPDataset1(project_dir=parameters.validation_folder,
                                                       transform=transforms.ToTensor())
        validate_loader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False)
        model_tools.validate(model, validate_loader, device)


if __name__ == "__main__":
    main()
