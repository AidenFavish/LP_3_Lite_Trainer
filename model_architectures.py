import torch.nn as nn
import torch.optim as optim
import torchvision.models

import parameters


class DefaultCNN(nn.Module):
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
        self.fc1 = nn.Linear(16 * self.pool_output_size * self.pool_output_size, 256)
        self.fc2 = nn.Linear(256, 4)

        # Activation functions
        self.act1 = nn.ReLU()
        self.act2 = nn.ELU()
        self.act3 = nn.Sigmoid()

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
        x = self.act2(self.fc1(x))
        x = self.act3(self.fc2(x))
        return x


class YOLONetwork(nn.Module):
    """
    Enhanced version of the YOLO architecture for object detection.
    It takes in 1000x1000 RGB images and outputs a 4-dimensional vector.
    The output vector represents the normalized center (x and y), width, and height of the detected object.

    Components:
    1. Convolutional layer: 3 channel input, 16 output channels, 3x3 kernel, stride 1, padding 1
    2. Convolutional layer: 16 channel input, 32 output channels, 3x3 kernel, stride 1, padding 1
    3. Convolutional layer: 32 channel input, 64 output channels, 3x3 kernel, stride 1, padding 1
    4. Max pooling layer: 2x2 kernel, stride 2
    5. Fully connected layer: Inputs from flattened conv output, 512 outputs
    6. Fully connected layer: 512 inputs, 4 outputs (normalized)
    7. ReLU activation for convolutional layers
    8. Sigmoid activation for final output normalization
    """
    def __init__(self):
        super(YOLONetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate output size after convolutions and pooling
        self.conv_output_size = (1000 - 3 + 2 * 1) // 1 + 1  # Update as per your network architecture
        self.pool_output_size = self.conv_output_size // 2

        # Fully connected layers
        self.fc1 = nn.Linear(64 * self.pool_output_size * self.pool_output_size, 512)
        self.fc2 = nn.Linear(512, 4)  # 4 outputs for normalized coordinates

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # To normalize the output between 0 and 1

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=parameters.learning_rate)

        # Loss Function
        self.loss_function = nn.MSELoss()  # Placeholder, YOLO uses a complex custom loss

    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(self.relu(self.conv3(x)))

        # Flattening for fully connected layers
        x = x.view(-1, 64 * self.pool_output_size * self.pool_output_size)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))  # Normalize the output

        return x


class EnhancedFasterRCNNNetwork(nn.Module):
    """
    Enhanced version of the Faster R-CNN architecture for object detection.
    This model utilizes a pre-trained ResNet-50 for feature extraction,
    followed by a Region Proposal Network (RPN) and ROI Pooling.
    It takes in 1000x1000 RGB images and outputs 4 normalized values (x, y, width, height)
    representing the bounding box of the detected object.

    Components:
    1. Feature Extractor: Pre-trained ResNet-50
    2. Region Proposal Network: Convolutional layer for region proposal
    3. ROI Pooling: To pool features in proposed regions (Implementation Placeholder)
    4. Fully connected layer: Classifier, 1024 outputs
    5. Fully connected layer: Bounding box regressor, 4 outputs (normalized)
    6. ReLU activation for intermediate layers
    7. Sigmoid activation for output normalization
    """
    def __init__(self):
        super(EnhancedFasterRCNNNetwork, self).__init__()

        # Load a pre-trained model for feature extraction
        self.feature_extractor = torchvision.models.resnet50()

        # Region Proposal Network (RPN)
        self.rpn = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Additional layers can be added here
        )

        # ROI Pooling
        # In practice, use operations like RoIPool or RoIAlign

        # Classifier and Bounding Box Regressor
        self.classifier = nn.Linear(512 * 7 * 7, 1024)
        self.bbox_regressor = nn.Linear(1024, 4)  # 4 outputs for normalized coordinates

        # Activation Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # To normalize the output between 0 and 1

        # Loss Function
        # Faster R-CNN uses a combination of classification loss and bounding box regression loss

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)

        # Region proposals
        proposals = self.rpn(x)

        # ROI pooling
        # Pooling features in the proposed regions

        # Classifier and regressor
        x = self.relu(self.classifier(proposals))
        bbox_deltas = self.sigmoid(self.bbox_regressor(x))  # Normalize the output

        return bbox_deltas


class MultiGPU_CNN(nn.Module):
    def __init__(self):
        super(MultiGPU_CNN, self).__init__()

        # Convolutional layers on GPU 0
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).to('cuda:0')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0).to('cuda:0')
        self.act1 = nn.ReLU().to('cuda:0')

        # Calculate output sizes
        self.conv_output_size = (1000 - 3 + 2 * 1) // 1 + 1
        self.pool_output_size = self.conv_output_size // 2

        # Fully connected layers each on different GPUs
        self.fc1 = nn.Linear(16 * self.pool_output_size * self.pool_output_size, 256).to('cuda:1')
        self.act2 = nn.ELU().to('cuda:1')
        self.fc2 = nn.Linear(256, 4).to('cuda:2')
        self.act3 = nn.Sigmoid().to('cuda:2')

        # Optimizer (needs careful handling in multi-GPU setup)
        self.optimizer = optim.Adam(self.parameters(), lr=parameters.learning_rate, weight_decay=parameters.weight_decay)

        # Loss function
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        # Convolutional part on GPU 0
        x = x.to('cuda:0')
        x = self.pool(self.act1(self.conv1(x)))

        # Transition to GPU 1 for first fully connected layer
        x = x.to('cuda:1')
        x = x.view(-1, 16 * self.pool_output_size * self.pool_output_size)
        x = self.act2(self.fc1(x))

        # Transition to GPU 2 for second fully connected layer
        x = x.to('cuda:2')
        x = self.act3(self.fc2(x))

        return x

