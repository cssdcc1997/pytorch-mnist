import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from torchvision.utils import make_grid
from matplotlib.pyplot import MultipleLocator

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #nn.init()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #show_graph(x, "origin")
        x = self.conv1(x)
        #show_graph(x, "conv1")
        x = F.relu(x) #f(x) = max(0, x)
        #show_graph(x, "conv1_relu")
        x = F.max_pool2d(x, 2)
        #show_graph(x, "conv1_relu_maxpool")
        x = self.conv2(x)
        #show_graph(x, "conv2")
        x = F.relu(x)
        #show_graph(x, "conv2_relu")
        x = F.max_pool2d(x, 2)
        #show_graph(x, "conv2_relu_maxpool")
        x = x.view(x.size(0), -1)
        #print(x[0].size())
        x = self.fc1(x)
        #print(x[0].size())
        x = F.relu(x)
        x = self.fc2(x)
        #print(x[0].size())
        x = F.relu(x)
        x = self.fc3(x)
        #print(x[0])
        x = F.log_softmax(x, dim=1)
        #print(x[0])
        #exit()

        return x

def show_graph(x, string):
    # y = copy.deepcopy(x[0][0])
    # make_grid(y)
    # plt.imshow(y.cpu().numpy(), cmap='gray')
    # plt.grid()
    # plt.show()
    y = copy.deepcopy(x[0])
    print(y[0])
    y = y * 0.3081 + 0.1307
    y = y.cpu().numpy()
    print(len(y[0]))
    print(y[0])
    ax = plt.gca()   
    if len(y[0]) < 10:
        x_major_locator=MultipleLocator(1)
        y_major_locator=MultipleLocator(1)
    else:
        x_major_locator=MultipleLocator(5)
        y_major_locator=MultipleLocator(5)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    for i in range(len(y)):
        plt.imshow(y[0], cmap='gray')
        
        #plt.grid(b=True, which="major", axis="both", ls="--")
        #plt.xlim(0, len(y[0]))
        #plt.ylim(len(y[0]), 0)
        plt.title("LeNet_{}_{}".format(string, i + 1))
        #plt.savefig("E:/WorkSpace/Pytorch/mnist/model/lenet_feature_map/{}_{}".format(string, i + 1))
        plt.show()
    
