import torch.nn as nn
import torch.optim as optim
import parameters

"""
This is a basic CNN architecture for processing images.
It takes in 1000x1000 RGB images and outputs a 4-dimensional vector.
The output vector is the predicted center (x and y), the radius of the square, and turn/tilt.

Components:
1. 3 channel input, 16 output channels, 3x3 kernel, stride 1, padding 1 conv layer
2. ReLU activation
3. 2x2 max pooling layer
4. Linear layer with 16 * 500 * 500 inputs (16 channels * pool_output_size * pool_output_size) and 512 outputs
5. TanH activation
6. Linear layer with 512 inputs and 128 outputs
7. ELU activation
8. Linear layer with 128 inputs and 4 outputs
"""
class DefaultCNN(nn.Module):
    def __init__(self):
        super(DefaultCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate output: conv_output = (input - kernel_size + 2 * padding) / stride + 1
        self.conv_output_size = (1000 - 3 + 2 * 1) // 1 + 1
        # Calculate pooling output size: pool_output = (conv_output - kernel_size) / stride + 1
        self.pool_output_size = self.conv_output_size // 2

        # Fully connected layers
        self.fc1 = nn.Linear(16 * self.pool_output_size * self.pool_output_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

        # Activation functions
        self.act1 = nn.ReLU()
        self.act2 = nn.ELU()
        self.act3 = nn.Tanh()

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=parameters.learning_rate, weight_decay=parameters.weight_decay)

        # Loss function
        self.loss_function = nn.MSELoss()

        # Version
        self.version = parameters.version

    def forward(self, x):
        # pass the input through the layers
        x = self.pool(self.act1(self.conv1(x)))
        x = x.view(-1, 16 * self.pool_output_size * self.pool_output_size)
        x = self.act3(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x
