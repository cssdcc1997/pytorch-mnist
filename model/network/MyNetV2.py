import torch
import torch.nn as nn
import torch.nn.functional as F

# parameters:
# 90 1800 7200 28800 28800 7680 480
# sum = 66690

class MyNetV2(nn.Module):
    def __init__(self):
        super(MyNetV2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 40, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 80, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 40, 3),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(    # dropout似乎用在分类的时候
            nn.Dropout(),
            nn.Linear(40 * 2 * 2, 48), # 8192
            nn.ReLU(inplace=True),
            nn.Linear(48, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)

        return x