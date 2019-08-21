Image Retrieve
=====
By Xiami2019
## Abstact  
This project is an Image Retrieve model for CUB_200_2011 and Stanford Dogs dataset and can attain state of the art performance.  

## Introduction  
Image Retrieve is a prevail task of computer vision.
It means you can input an image as a query into the model, then the model will recall some images which are similar with the query one and are ranked by the similarity from high to low.
This task has many usage scenarios, such as Taobao's PaiLiTao and Google's image search etc. For example, you can take a photo of what you want to buy, and search the products with your's photo.

## Learning to Hash  
As we use image feature map extracted by Convolutional Neural Networks, for the sake of saving image and calculating the similarity, we change the image feature maps to binary hash codes. So, we can use L2 or L1 norm to calculate the images' similarity efficiently. We usually use a activation function tanh or sigmoid to restrict the model's output. By the way, during training we use fraction to calculate loss value and during testing we change the output to bianry hash code.

## Triplet Loss  
<a href="https://www.codecogs.com/eqnedit.php?latex=L_{triplet}(F(I),F(I^{&plus;}),F(I^{-}))&space;\\&space;=&space;max(0,\left&space;\|&space;F(I)-F(I^{&plus;})&space;\right&space;\|_{2}^{2}-\left&space;\|F(I)-F(I^{-})&space;\right&space;\|_{2}^{2}&plus;margin)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_{triplet}(F(I),F(I^{&plus;}),F(I^{-}))&space;\\&space;=&space;max(0,\left&space;\|&space;F(I)-F(I^{&plus;})&space;\right&space;\|_{2}^{2}-\left&space;\|F(I)-F(I^{-})&space;\right&space;\|_{2}^{2}&plus;margin)" title="L_{triplet}(F(I),F(I^{+}),F(I^{-})) \\ = max(0,\left \| F(I)-F(I^{+}) \right \|_{2}^{2}-\left \|F(I)-F(I^{-}) \right \|_{2}^{2}+margin)" /></a>  
Margin is hyperparameter determined by the hash code size.  
Triplet loss is able to decrease the distance between the similar images and increase the distance between dissimilar images.

## Metric  
In image retrieve task, we often use mAP(mean average precision) or precision/recall rate to estimate the model's performance. In this project I use map as the metric.

## Requirements
Python 3.6  
Pytorch 1.1.0  
 
## Implemention Detials
This project use a pretrained Resnet18 changed the last full connection layer to fine-tune and Adam_inverse_sqrt algorithm to optimize.   
I use the online methods to get triplets and also online methods to calculate **mAP**.  
At the model's output, I use a tanh activation function to restrict the output to [-1, 1].  
When convert the model's output to binary hash code, I use `-1` to replace the code `0` and simply convert all the negative value to `-1` and all the positive value to `1`.  
I use Euclidean distance to present the similarities between images.  
The optimizer part is and the logger part is released by FAIR in their project https://github.com/facebookresearch/XLM.
The optimizer is **AdamInverseSqrtWithWarmup**, which can decay the LR based on the inverse square root of the update number and also support a warmup phase.  
For retrieve task, following the official setting, I use the train set as the retrieve database and the test set as query images.  


## Quick Start
At first you should download the datasets by your self and unzip the datasets at correct position. Also I will later add a shell script to download datasets automaticly.
Then you cun run the command:  
```Bash  
python train.py --exp_name 'CUB_32' --code_size 32 --triplet_margin 8 --dataset_name 'CUB_200_2011'
```  
## Result
Got results of `mAP` on two datasets after about 1000~4000 epochs training(Best model appeared at different epoch for different code size):  

Datasets | 16bits | 32bits | 48bits | 64bits
|:---: |:---: |:---: | :---: |:---: |
`CUB_200_2011` | **0.6560** | **0.6911** | **0.6941** | **0.7047**
`CUB Baseline` | 0.5173 | 0.6518 | 0.6807 | 0.6949
`Stanford Dogs` | **0.6883** | 0.7049 | 0.7106 | 0.7123
`Dogs Baseline` | 0.6745 | **0.7101** | **0.7252** | **0.7293**
## Reference
[1] Hanjiang Lai, Yan Pan, Ye Liu, Shuicheng Yan [*Simultaneous Feature Learning and Hash Coding with Deep Neural Networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Lai_Simultaneous_Feature_Learning_2015_CVPR_paper.html)  
[2] Hanjiang Lai, Jikai Chen, Libing Geng, Yan Pan, Xiaodan Liang, Jian Yin [*Improving Deep Binary Embedding Networks by
Order-aware Reweighting of Triplets*](https://ieeexplore.ieee.org/abstract/document/8640819)
