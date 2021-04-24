import torch
import torch.nn as nn
import torch.nn.functional as F

class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.features = nn.Sequential(
            # 28*28
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            
            ## 14*14
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            nn.Dropout(0.2),

            nn.Conv2d(16, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            
            ## 7*7
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            
            ## 4*4
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.Dropout(0.2),

            ## down sampling
            ## 2*2
            nn.Conv2d(128, 96, 3, 2, 1),
            nn.BatchNorm2d(96),
            nn.PReLU(96),
            nn.Dropout(0.2),
            
            ## 1*1
            nn.Conv2d(96, 10, 3, 2, 1),
            nn.BatchNorm2d(10),
            nn.PReLU(10),
            nn.Dropout(0.2)
        )
        
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=1)

        return x