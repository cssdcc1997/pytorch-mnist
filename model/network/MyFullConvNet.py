import torch
import torch.nn as nn
import torch.nn.functional as F
# sum parameter = 75280

class MyFullConvNet(nn.Module):
    def __init__(self):
        super(MyFullConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 24, 3)
        self.conv4 = nn.Conv2d(24, 48, 3)
        self.conv5 = nn.Conv2d(48, 96, 3)
        self.conv6 = nn.Conv2d(96, 48, 2)
        self.fc1 = nn.Linear(48, 10)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        

        return x
