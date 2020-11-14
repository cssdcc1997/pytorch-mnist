# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt



# Mnist digits dataset
# if not(os.path.exists('./data/')) or not os.listdir('./MINIST/'):
#     # not mnist dir or mnist is empyt dir
#     DOWNLOAD_MNIST = True
 
train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    #download=DOWNLOAD_MNIST,
)
 
# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.targets.size())               # (60000)



for i in range(0, 9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_data.train_data[i].numpy(), cmap='gray')

print(train_data.train_data[0])
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()