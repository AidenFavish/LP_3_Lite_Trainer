from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir=runs
import parameters


# Trains the model
def train(model, dataloader, device):
    print('Starting Training...')

    # Setup
    writer = SummaryWriter(f"runs/{parameters.version}/{parameters.run_instance}")  # Tensorboard writer
    num_epochs = parameters.num_epochs

    optimizer = model.optimizer
    criterion = model.loss_function

    for epoch in range(num_epochs):
        print("------------------------------------------")

        running_loss = 0.0  # Loss for the epoch
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
            if i % parameters.batch_print == 0:
                print(f'[{epoch + 1}, {i + 1}] avg loss: {avg_loss:.3f}')

        print(f"------------------------------------------\n\033[1;31mEpoch {epoch + 1} finished. Average Loss: {running_loss/len(dataloader)}\033[0m\n")


def train2(model, dataloader):
    print('Starting Training...')

    # Setup
    writer = SummaryWriter(f"runs/{parameters.version}/{parameters.run_instance}")  # Tensorboard writer
    num_epochs = parameters.num_epochs

    # Assuming the optimizer and criterion are defined as part of the model
    optimizer = model.optimizer
    criterion = model.loss_function

    for epoch in range(num_epochs):
        print("------------------------------------------")

        running_loss = 0.0  # Loss for the epoch
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            # Move inputs and labels to the first device (cuda:0)
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:2')  # Assuming the last layer is on cuda:2

            optimizer.zero_grad()

            # Forward pass (model handles device transfers internally)
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()

            # Step optimizer
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Calculate average loss
            avg_loss = running_loss / (i + 1)
            writer.add_scalar('training loss', avg_loss, epoch * len(dataloader) + i)
            if i % parameters.batch_print == 0:
                print(f'[{epoch + 1}, {i + 1}] avg loss: {avg_loss:.3f}')

        print(
            f"------------------------------------------\n\033[1;31mEpoch {epoch + 1} finished. Average Loss: {running_loss / len(dataloader)}\033[0m\n")

