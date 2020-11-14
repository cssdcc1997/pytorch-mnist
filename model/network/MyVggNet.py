import torch
import torch.nn as nn
import torch.nn.functional as F

class MyVggNet(nn.Module):
    def __init__(self):
        super(MyVggNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        
            nn.Conv2d(16, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 10)
        )
        
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)

        return x