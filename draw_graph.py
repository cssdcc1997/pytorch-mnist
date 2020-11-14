import matplotlib
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="lenet")
args = parser.parse_args()

#file_loss_path = "E:/WorkSpace/Pytorch/mnist/model/result/{}_loss.txt".format(args.model)
file_loss_path = sys.path[0] + "/model/result/{}_loss.txt".format(args.model)

lst_loss = list()
with open(file_loss_path) as file_object:
    for line in file_object:
        if "e" in line:
            lst_loss.append(eval(line))
        else:
            lst_loss.append(float(line[:-2]))
    file_object.close()

#file_acc_path = "E:/WorkSpace/Pytorch/mnist/model/result/{}_acc.txt".format(args.model)
file_acc_path = sys.path[0] +  "/model/result/{}_acc.txt".format(args.model)
lst_acc = list()
with open(file_acc_path) as file_object:
    for line in file_object:
        if "e" in line:
            lst_acc.append(eval(line))
        else:
            lst_acc.append(float(line[:-2]))
    file_object.close()
print(lst_acc)

plt.title("{} loss".format(args.model))
plt.plot(lst_loss)
plt.xlim(0 - len(lst_loss) / 20, len(lst_loss))
plt.ylim(0, 1.5)
plt.grid()
plt.savefig(file_loss_path[:-3] + "jpg")

plt.title("{} acc".format(args.model))
plt.plot(lst_acc)
plt.xlim(0 - len(lst_acc) / 20, len(lst_acc))
plt.ylim(min(lst_acc) - 1, max(max(lst_acc) + 1, 100))
plt.savefig(file_acc_path[:-3] + "jpg")