# pytorch-mnist

<img src="https://github.com/cssdcc1997/pytorch-mnist/blob/main/model/lenet_feature_map/origin_1.png" width=400 />

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

| network | accuracy(%)|
|---|---|
|LeNet|98.91|
|MyFullConvNet|98.06|
|MyNetV1|99.19|
|MyNetV2|98.81|
|**`MyVggNet`**|99.27|

# feature map visualization

The feature map is generated during the test process of LeNet.

## origin graph

<img src="./model/lenet_feature_map/origin_1.png" width=400 />

The image above is the origin image of the MNIST dataset -num 8. Pixel size is `28 x 28`.

## conv1
<table>
    <tr>
        <img src="./model/lenet_feature_map/conv1_1.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_2.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_3.png" width=200 />
    </tr>
    <tr>
        <img src="./model/lenet_feature_map/conv1_4.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_5.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_6.png" width=200 />
    </tr>
</table>

After once convolution, we can still recognize the image as a number `8`. Because of the output_channel number of the first convolution layer, there `6 feature maps`.

## conv1_relu
<table>
    <tr>
        <img src="./model/lenet_feature_map/conv1_relu_1.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_relu_2.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_relu_3.png" width=200 />
    </tr>
    <tr>
        <img src="./model/lenet_feature_map/conv1_relu_4.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_relu_5.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_relu_6.png" width=200 />
    </tr>
</table>


## conv1_relu_maxpool
<table>
    <tr>
        <img src="./model/lenet_feature_map/conv1_relu_maxpool_1.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_relu_maxpool_2.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_relu_maxpool_3.png" width=200 />
    </tr>
    <tr>
        <img src="./model/lenet_feature_map/conv1_relu_maxpool_4.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_relu_maxpool_5.png" width=200 />
        <img src="./model/lenet_feature_map/conv1_relu_maxpool_6.png" width=200 />
    </tr>
</table>