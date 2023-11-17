import torch
import torch.nn as nn
import torchvision.models.resnet as torch_resnet
from torch.hub import load_state_dict_from_url

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Layer 1: Initial layer with a 17x17 filter
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(17,17), padding=8, padding_mode='reflect')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Additional layer with an 11x11 filter
        self.conv2 = nn.Conv2d(32, 32, kernel_size=11, padding=5, padding_mode='reflect')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Intermediate layers with 5x5 filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, padding_mode='reflect')
        self.relu3 = nn.ReLU()
        
        # Layer 4: Another layer with 5x5 filter and adjusted stride
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=2, padding_mode='reflect')
        self.relu4 = nn.ReLU()
        
        # Layer 5: Final layer with 5x5 filter
        self.conv5 = nn.Conv2d(64, 64, kernel_size=5, padding=2, padding_mode='reflect')
        self.relu5 = nn.ReLU()

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

    def forward(self, x):
        # Forward pass through the layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        return x

class CustomCNN2(nn.Module):
    def __init__(self):
        super(CustomCNN2, self).__init__()
        
        # Layer 1: Initial layer with a 17x17 filter
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(17,17), padding=8, padding_mode='reflect')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Additional layer with an 11x11 filter
        self.conv2 = nn.Conv2d(4, 16, kernel_size=11, padding=5, padding_mode='reflect')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Intermediate layers with 5x5 filters
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, padding=2, padding_mode='reflect')
        self.relu3 = nn.ReLU()
        
        # Layer 4: Another layer with 5x5 filter and adjusted stride
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, padding=2, padding_mode='reflect')
        self.relu4 = nn.ReLU()
        
        # Layer 5: Final layer with 5x5 filter
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2, padding_mode='reflect')
        self.relu5 = nn.ReLU()

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

    def forward(self, x):
        # Forward pass through the layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        return x

class CustomCNN3(nn.Module):
    def __init__(self):
        super(CustomCNN3, self).__init__()
        
        # Layer 1: Initial layer with a 17x17 filter
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(17,17), padding=8, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(4)
        self.ln1 = nn.LayerNorm((4,400,48))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Additional layer with an 11x11 filter
        self.conv2 = nn.Conv2d(4, 16, kernel_size=11, padding=5, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(16)
        self.ln2 = nn.LayerNorm((16,200,24))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Intermediate layers with 5x5 filters
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, padding=2, padding_mode='reflect')
        self.bn3 = nn.BatchNorm2d(16)
        self.ln3 = nn.LayerNorm((16,100,12))
        self.relu3 = nn.ReLU()
        
        # Layer 4: Another layer with 5x5 filter and adjusted stride
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, padding=2, padding_mode='reflect')
        self.bn4 = nn.BatchNorm2d(32)
        self.ln4 = nn.LayerNorm((32,100,12))
        self.relu4 = nn.ReLU()
        
        # Layer 5: Final layer with 5x5 filter
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2, padding_mode='reflect')
        self.bn5 = nn.BatchNorm2d(32)
        self.ln5 = nn.LayerNorm((32,100,12))
        self.relu5 = nn.ReLU()

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

    def forward(self, x):
        # Forward pass through the layers
        x = self.pool1(self.relu1(self.ln1(self.conv1(x))))
        x = self.pool2(self.relu2(self.ln2(self.conv2(x))))
        x = self.relu3(self.ln3(self.conv3(x)))
        x = self.relu4(self.ln4(self.conv4(x)))
        x = self.relu5(self.ln5(self.conv5(x)))
        return x