import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class mynn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 16, 3)
        self.conv3 = torch.nn.Conv2d(16, 16, 3)
        self.linear = torch.nn.Linear(16*48*48, 128)
        self.linear2 = torch.nn.Linear(128, 32)
        self.linear3 = torch.nn.Linear(32, 4)
        self.soft = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.maxpool= torch.nn.MaxPool2d(2)
     
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = x.reshape(-1,16*48*48)
        x = self.relu(self.linear(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x