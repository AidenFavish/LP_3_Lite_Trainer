import torch.nn as nn
import torch.optim as optim
import torchvision.models
import torch
import parameters


class ImprovedCNN(nn.Module):
    """
    Improved CNN architecture for object detection.
    This model inputs 1000x1000 RGB images and outputs a 4-dimensional vector.
    The output vector represents the normalized center (x and y), width, and height of the detected object.
    """

    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

        # Calculate output sizes for conv and pooling layers
        self.conv_output_size1 = ((1000 - 3 + 2 * 1) // 2 + 1)  # Output size after conv1
        self.conv_output_size2 = (((self.conv_output_size1 * self.conv_output_size1 // 4) - 3 + 2 * 1) // 2 + 1)  # Output size after conv2
        self.pool_output_size = 62  # magic number

        # Fully connected layers
        self.fc1 = nn.Linear(32 * self.pool_output_size * self.pool_output_size, 42)
        self.fc2 = nn.Linear(42, 4)

        # Activation functions
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=parameters.learning_rate)

        # Loss function
        self.loss_function = PFLoss()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act1(self.pool(x))
        x = self.act1(self.conv2(x))
        x = self.act1(self.pool(x))
        x = self.dropout(x)
        x = x.view(-1, 32 * self.pool_output_size * self.pool_output_size)
        x = self.act2(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x


class IdentityNetwork(nn.Module):
    """
    Custom CNN architecture.
    Sequence: Convolution -> ReLU -> MaxPool -> Convolution -> Identity Block ->
              Linear -> ReLU -> Linear -> Sigmoid.
    Outputs a 4-dimensional vector.
    """

    def __init__(self):
        super(IdentityNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Identity Block
        self.id_block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1)
        )

        # Calculate the size of the flattened layer before the first linear layer
        self._to_linear = None
        self._calculate_to_linear(1000, 1000)  # Assuming input size is 1000x1000

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 4)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=parameters.learning_rate)

        # Loss function
        self.loss_function = nn.MSELoss()  # Placeholder loss function, replace as needed

    def _calculate_to_linear(self, h, w):
        x = torch.randn(3, h, w).unsqueeze(0)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.conv2(x)
        identity = x
        x = self.id_block(x)
        x += identity
        x = self.pool(x)  # Assuming another pooling layer here
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv2(x)
        identity = x
        x = self.id_block(x)
        x += identity
        x = self.pool(x)  # Assuming another pooling layer here
        x = x.view(-1, self._to_linear)  # Flatten the output for the linear layer
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class PFLoss(nn.Module):
    def __init__(self, diversity_factor=1.0, min_std_dev=0.3):
        super(PFLoss, self).__init__()
        self.diversity_factor = diversity_factor
        self.min_std_dev = min_std_dev

    def forward(self, predictions, targets):
        # Compute the standard deviation of the predictions
        std_dev = predictions.std(dim=0)

        # Calculate the diversity penalty
        # The penalty is applied if the standard deviation is lower than the threshold
        diversity_penalty = torch.where(std_dev < self.min_std_dev, 1 + 1 / torch.abs(std_dev), torch.zeros_like(std_dev)).mean()
        components = [10 * torch.mean(10 ** torch.abs(predictions - targets) - 1), 0.3 * torch.mean(diversity_penalty)]

        loss = components[0] + components[1]
        return loss, components
