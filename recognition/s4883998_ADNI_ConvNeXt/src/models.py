
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.linear = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d(),

            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d(),

            
        )

    def forward(self, x):
        y = self.linear(x)
        
        