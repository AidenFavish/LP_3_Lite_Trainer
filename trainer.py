from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir=runs --host 192.168.4.247 --port 8080
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch
import model_tools
import parameters
import time


# Trains the model
def train(model, dataloader, device):
    print('Starting Standard Training...')

    # Setup
    writer = SummaryWriter(f"runs/{parameters.version}/{parameters.run_instance}")  # Tensorboard writer
    num_epochs = parameters.num_epochs

    # Get the optimizer and loss function
    if parameters.parallel_processing:
        optimizer = model.module.optimizer
        criterion = model.module.loss_function
    else:
        optimizer = model.optimizer
        criterion = model.loss_function

    for epoch in range(num_epochs):
        print("------------------------------------------")

        start_time = time.time()
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
            if i == len(dataloader) - 1:
                print(
                    f'[{epoch + 1}, {i + 1}] avg loss: {avg_loss:.3f}\nEpoch mini-test:\npredict: {outputs.tolist()[0]}\nactual: {labels.tolist()[0]}')
            elif i % parameters.batch_print == 0:
                print(f'[{epoch + 1}, {i + 1}] avg loss: {avg_loss:.3f}')

        duration = time.time() - start_time
        print(
            f"------------------------------------------\n\033[1;31mEpoch {epoch + 1} finished. Average Loss: {running_loss / len(dataloader)}\033[0m")
        seconds = int(duration * (num_epochs - epoch - 1))
        print(f"\033[1;33mETA: {seconds // 60} minutes and {seconds % 60} seconds\033[0m\n")


# for the multi trainer
model = None
train_loader = None


def mp_train_helper(rank, world_size):
    print("Starting Multi-Processing Training...")

    # Setup
    writer = SummaryWriter(f"runs/{parameters.version}/{parameters.run_instance}")  # Tensorboard writer
    model_tools.setup(rank, world_size)
    num_epochs = parameters.num_epochs

    # Create model, optimizer, and dataloader
    global model  # Your model
    optimizer = model.optimizer  # Your optimizer
    criterion = model.loss_function  # Your loss function
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    global train_loader  # Your dataset
    sampler = DistributedSampler(train_loader, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_loader, batch_size=..., sampler=sampler)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        print("------------------------------------------")

        start_time = time.time()
        running_loss = 0.0  # Loss for the epoch
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            avg_loss = running_loss / (i + 1)  # Average loss for the epoch so far
            writer.add_scalar('training loss', avg_loss, epoch * len(dataloader) + i)
            if i == len(dataloader) - 1:
                print(
                    f'[{epoch + 1}, {i + 1}] avg loss: {avg_loss:.3f}\nEpoch mini-test:\npredict: {outputs.tolist()[0]}\nactual: {labels.tolist()[0]}')
            elif i % parameters.batch_print == 0:
                print(f'[{epoch + 1}, {i + 1}] avg loss: {avg_loss:.3f}')

        duration = time.time() - start_time
        print(
            f"------------------------------------------\n\033[1;31mEpoch {epoch + 1} finished. Average Loss: {running_loss / len(dataloader)}\033[0m")
        seconds = int(duration * (num_epochs - epoch - 1))
        print(f"\033[1;33mETA: {seconds // 60} minutes and {seconds % 60} seconds\033[0m\n")


def mp_train(model1, train_loader1):
    global model
    global train_loader
    model = model1
    train_loader = train_loader1
    world_size = torch.cuda.device_count()
    mp.spawn(mp_train_helper, args=(world_size,), nprocs=world_size, join=True)
