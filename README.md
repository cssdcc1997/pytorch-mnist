# pytorch-mnist

![](https://github.com/cssdcc1997/pytorch-mnist/blob/main/model/lenet_feature_map/origin_1.png)

This program realize the recognization function of `MNIST` data. The original edition is [https://github.com/pytorch/examples/blob/master/mnist/main.py](https://github.com/pytorch/examples/blob/master/mnist/main.py).

In addtion to basic recognization function, i add some data visualization item including **training results visualization** and **feature map visualization**. 

# traning results
```cmd
python train.py --model LeNet
```

Enter the above script to start training.

I designed a few networks by myself which contain some of the advantages of classical neural networks such as: AlexNet, VggNet and so on. You can try others networks to see which works better. My conclusion is that the **deeper is better**. The `MyVggNet`, which gives the best results, is in imitation of VggNet.

```cmd
python train.py --model MyVggNet
```

![](https://github.com/cssdcc1997/pytorch-mnist/blob/main/model/result/Figure_1.png)

