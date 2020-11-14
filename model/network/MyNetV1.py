import torch
import torch.nn as nn
import torch.nn.functional as F

# Designer: lyf
# It attain quiet great results. The accuracy is 99.36%
# Design idea:
#     use small convolution kernel as much as i can
#     small convolution kernel (3 * 3) can reduce the parameters nums
#     And i use convolution with stride 2 to replace the max pooling
#     In the classifier, i use dropout to prevent the overfitting
#     In addition, i choose the input channel and output channel carefully
#     The channel should be a multiple or power of 2. 

class MyNetV1(nn.Module):
    def __init__(self):
        super(MyNetV1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 48, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 3, 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(    # dropout似乎用在分类的时候
            nn.Dropout(),
            nn.Linear(512 * 4 * 4, 1024), # 8192
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)

        return x