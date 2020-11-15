from __future__ import print_function # 这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse
import os
#import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time

# import network
from model.network.LeNet import LeNet
from model.network.MyNetV1 import MyNetV1
from model.network.MyNetV2 import MyNetV2
from model.network.DefaultNet import DefaultNet
from model.network.MyFullConvNet import MyFullConvNet
from model.network.MyVggNet import MyVggNet

graph_loss = []
graph_acc = []

def train(args, model, device, train_loader, optimizer, epoch):
    # 这里的train和上面的train不是一个train
    model.train()
    start_time = time.time()
    tmp_time = start_time
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()   # 优化器梯度为什么初始化为0？
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Cost time: {:.6f}s".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), time.time() - tmp_time
            ))
            tmp_time = time.time()
            graph_loss.append(loss.item())
            if args.dry_run:
                break
    end_time = time.time()
    print("Epoch {} cost {} s".format(epoch, end_time - start_time))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss 
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        float(100. * correct / len(test_loader.dataset))
    ))

    graph_acc.append(100. * correct / len(test_loader.dataset))

# action 和 gamma , metavar的作用
def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default : 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default : 1000)")
    parser.add_argument("--epochs", type=int, default=64, metavar="N",
                        help="number of epochs to train (default : 64)")
    parser.add_argument("--learning-rate", type=float, default=0.1, metavar="LR",
                        help="number of epochs to train (default : 14)")
    parser.add_argument("--gamma", type=float, default=0.5, metavar="M",
                        help="Learning rate step gamma (default : 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=True,
                        help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default : 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action = "store_true", default=True,
                        help="For saving the current Model")
    parser.add_argument("--load_state_dict", type=str, default="no",
                        help="load the trained model weights or not (default: no)")
    parser.add_argument("--model", type=str, default="LeNet",
                        help="choose the model to train (default: LeNet)")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available() # not > and > or
    print("user cuda is {}".format(use_cuda))
    torch.manual_seed(args.seed)    # 设置随机种子，什么是随机种子？

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),          # to tensor, why?
        # normalize(mean, std, inplace=False) mean各通道的均值， std各通道的标准差， inplace是否原地操作
        # 这里说的均值是数据里的均值
        # output = (input - mean) / std
        # 归一化到-1 ~ 1，也不一定，但是属于标准化
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset1 = datasets.MNIST("./data", train=True, download=True,
                            transform=transform)
    dataset2 = datasets.MNIST("./data", train=False,
                            transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model_name = args.model.lower()
    if model_name == "lenet":
        model = LeNet().to(device)
    elif model_name == "defaultnet":
        model = DefaultNet().to(device)
    elif model_name == "mynetv1":
        model = MyNetV1().to(device)
    elif model_name == "mynetv2":
        model = MyNetV2().to(device)
    elif model_name == "myfullconvnet":
        model = MyFullConvNet().to(device)
    elif model_name == "myvggnet":
        model = MyVggNet().to(device)



    #model = Net().to(device)
    model_path = Path("./model/weights/{}.pt".format(model_name))
    if model_path.exists() and args.load_state_dict == "yes":
        model.load_state_dict(torch.load(model_path))
        print("Load the last trained model.")
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    #optimizer_path = Path("./model/weights/")

    # scheduler是学习率调整，有lambdaLR机制和stepLR机制，lr = lr * gamma^n, n = epoch/step_size
    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./model/weights/{}.pt".format(model_name))

    # record the training results
    create_loss_txt_path = "./model/result/{}_loss.txt".format(model_name)
    create_acc_txt_path = "./model/result/{}_acc.txt".format(model_name)
    f = open(create_loss_txt_path, "w+")
    for loss in graph_loss: 
        f.writelines("{}\n".format(loss))
    f.close()
    f = open(create_acc_txt_path, "w+")
    for acc in graph_acc:
        f.writelines("{}\n".format(acc))
    f.close()


if __name__ == "__main__":
    main()