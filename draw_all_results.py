import matplotlib
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys

#model_name = ["lenet", "defaultnet", "mynetv1", "mynetv2", "myfullconvnet", "myvggnet"]
model_name = ["lenet", "mynetv1", "myvggnet"]

# lenet
lst_acc_lenet = list()
file_acc_path = sys.path[0] + "/model/result/{}_acc.txt".format("lenet")
with open(file_acc_path) as file_object:
    for line in file_object:
        if "e" in line:
            lst_acc_lenet.append(eval(line))
        else:
            lst_acc_lenet.append(float(line[:-2]))
    file_object.close()
plt.plot(lst_acc_lenet, label="{}".format("lenet"))

########################## defaultnet #########################
# lst_acc_defaultnet = list()
# file_acc_path = sys.path[0] + "/model/result/{}_acc.txt".format("defaultnet")
# with open(file_acc_path) as file_object:
#     for line in file_object:
#         if "e" in line:
#             lst_acc_defaultnet.append(eval(line))
#         else:
#             lst_acc_defaultnet.append(float(line[:-2]))
#     file_object.close()
# plt.plot(lst_acc_defaultnet, label="{}".format("defaultnet"))

############################ mynetv1 ######################
lst_acc_mynetv1 = list()
file_acc_path = sys.path[0] + "/model/result/{}_acc.txt".format("mynetv1")
with open(file_acc_path) as file_object:
    for line in file_object:
        if "e" in line:
            lst_acc_mynetv1.append(eval(line))
        else:
            lst_acc_mynetv1.append(float(line[:-2]))
    file_object.close()
plt.plot(lst_acc_mynetv1, label="{}".format("mynetv1"))

# ############################# mynetv2 ####################
# lst_acc_mynetv2 = list()
# file_acc_path = sys.path[0] + "/model/result/{}_acc.txt".format("mynetv2")
# with open(file_acc_path) as file_object:
#     for line in file_object:
#         if "e" in line:
#             lst_acc_mynetv2.append(eval(line))
#         else:
#             lst_acc_mynetv2.append(float(line[:-2]))
#     file_object.close()
# plt.plot(lst_acc_mynetv2, label="{}".format("mynetv2"))

# ######################### myfullconvnet ######################
# lst_acc_myfullconvnet = list()
# file_acc_path = sys.path[0] + "/model/result/{}_acc.txt".format("myfullconvnet")
# with open(file_acc_path) as file_object:
#     for line in file_object:
#         if "e" in line:
#             lst_acc_myfullconvnet.append(eval(line))
#         else:
#             lst_acc_myfullconvnet.append(float(line[:-2]))
#     file_object.close()
# plt.plot(lst_acc_myfullconvnet, label="{}".format("myfullconvnet"))

######################## myvggnet ###############################
lst_acc_myvggnet = list()
file_acc_path = sys.path[0] + "/model/result/{}_acc.txt".format("myvggnet")
with open(file_acc_path) as file_object:
    for line in file_object:
        if "e" in line:
            lst_acc_myvggnet.append(eval(line))
        else:
            lst_acc_myvggnet.append(float(line[:-2]))
    file_object.close()
plt.plot(lst_acc_myvggnet, label="{}".format("myvggnet"))

plt.grid()
plt.legend()
plt.show()