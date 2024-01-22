import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
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

        # Calculate output sizes for conv and pooling layers
        self.conv_output_size1 = ((1000 - 3 + 2 * 1) // 2 + 1)  # Output size after conv1
        self.conv_output_size2 = (((
                                           self.conv_output_size1 * self.conv_output_size1 // 4) - 3 + 2 * 1) // 2 + 1)  # Output size after conv2
        self.pool_output_size = 62  # magic number

        # Fully connected layers
        self.fc1 = nn.Linear(32 * self.pool_output_size * self.pool_output_size + 16 * 250 * 250, 42)
        self.fc2 = nn.Linear(42, 4)

        # Activation functions
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

        # Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=parameters.learning_rate, weight_decay=parameters.weight_decay)

        # Loss function
        self.loss_function = PFLoss()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x2 = self.act1(self.pool(x))
        x = self.act1(self.conv2(x2))
        x = self.act1(self.pool(x))
        x = x.view(-1, 32 * self.pool_output_size * self.pool_output_size)
        x2 = x2.view(-1, 16 * 250 * 250)
        x = self.act2(self.fc1(torch.cat((x, x2), dim=1)))
        x = self.act2(self.fc2(x))
        return x


class PFLoss(nn.Module):
    def __init__(self, diversity_factor=1.0, min_std_dev=0.25):
        super(PFLoss, self).__init__()
        self.diversity_factor = diversity_factor
        self.min_std_dev = min_std_dev

    def forward(self, predictions, targets):
        # Calculate the diversity penalty
        # The penalty is applied if the standard deviation is lower than the threshold
        xp = predictions[:, :1]
        yp = predictions[:, 1:2]
        rp = predictions[:, 2:3]
        xt = targets[:, :1]
        yt = targets[:, 1:2]
        rt = targets[:, 2:3]
        xpstd = torch.std(xp)
        ypstd = torch.std(yp)
        rpstd = torch.std(rp)
        xtstd = torch.std(xt)
        ytstd = torch.std(yt)
        rtstd = torch.std(rt)

        lossMSE = 3 ** torch.mean(3 ** torch.abs(predictions - targets) - 1, dim=0) - 1
        lossSTD = (2 ** torch.abs(xpstd - xtstd) - 1) + (2 ** torch.abs(ypstd - ytstd) - 1) + (
                2 ** torch.abs(rpstd - rtstd) - 1)  # Intentionally partially below y = x
        loss = lossMSE.sum() + 25 * lossSTD

        return loss, [lossMSE.tolist(), 25 * lossSTD.item()]


class ImprovedCNN2(nn.Module):
    def __init__(self):
        super(ImprovedCNN2, self).__init__()

        # Load a pre-trained ResNet model
        self.pretrained_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Optionally, you can freeze the layers of the pre-trained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer of ResNet for your specific task
        # For example, if the ResNet model outputs a feature vector of size 512,
        # and you want to output a 4-dimensional vector for your task:
        # self.pretrained_model.fc = nn.Linear(2048, 64)

        # Fully connected layers
        self.fc1 = nn.Linear(1000, 2)

        # Activation functions
        self.act1 = nn.Softmax(dim=0)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=parameters.learning_rate,
                                    weight_decay=parameters.weight_decay)

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.act1(self.fc1(x))
        return x
