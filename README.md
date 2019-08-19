Image Retrieve
=====
# Abstact  
This project is an Image Retrieve model for CUB_200_2011 and Stanford Dogs dataset and can attain state of the art performance.  
The author is XiaMi2019

# Introduction  
Image Retrieve is a prevail task of computer vision.
It means when you input an image as a query, model will recall images which are similar with the query one and are ranked with similarities from high to low.
This task can be uesd in many circumstances, such as Taobao's PaiLiTao. You can take a photo of what you want to buy, and search the products with your's photo.

# Learning to Hash  
As we use image feature map extracted by Convolutional Neural Networks, for the sake of saving image and calculating the similarity, we change the image feature maps to binary hash codes. So, we can use L2 or L1 norm to calculate the images' similarity efficiently. We usually use a activation function tanh or sigmoid to restrict the model's output. By the way, during training we use fraction to calculate loss value and during testing we change the output to bianry hash code.

# Triplet Loss  
<a href="https://www.codecogs.com/eqnedit.php?latex=L_{triplet}(F(I),F(I^{&plus;}),F(I^{-}))&space;\\&space;=&space;max(0,\left&space;\|&space;F(I)-F(I^{&plus;})&space;\right&space;\|_{2}^{2}-\left&space;\|F(I)-F(I^{-})&space;\right&space;\|_{2}^{2}&plus;margin)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_{triplet}(F(I),F(I^{&plus;}),F(I^{-}))&space;\\&space;=&space;max(0,\left&space;\|&space;F(I)-F(I^{&plus;})&space;\right&space;\|_{2}^{2}-\left&space;\|F(I)-F(I^{-})&space;\right&space;\|_{2}^{2}&plus;margin)" title="L_{triplet}(F(I),F(I^{+}),F(I^{-})) \\ = max(0,\left \| F(I)-F(I^{+}) \right \|_{2}^{2}-\left \|F(I)-F(I^{-}) \right \|_{2}^{2}+margin)" /></a>  
Margin is hyperparameter determined by the hash code size.  
Triplet loss is able to decrease the distance between the similar images and increase the distance between dissimilar images.

# Metric  
In image retrieve task, we often use mAP(mean average precision) or precision/recall rate to estimate the model's performance. Here we use map as our metric.

# Requirements
Python 3.6  
Pytorch 1.1.0  
TensorboardX  
 
# Implemention
This project use a pretrained Resnet18 to fine-tune.
The optimizer part and log part is released by FAIR in their project https://github.com/facebookresearch/XLM.

# Quick Start
At first you should download the datasets by your self and unzip the datasets at correct position. Also I will later add a shell script to download datasets automaticly.
Then you cun run the command:  
```Bash  
python train.py --exp_name 'CUB_32' --code_size 32 --triplet_margin 8 --dataset_name 'CUB_200_2011'
```  
# Result
We can get results of two datasets after about 2000 epochs training:  
Name | Academy | score 
- | :-: | -: 
Harry Potter | Gryffindor| 90 
Hermione Granger | Gryffindor | 100 
Draco Malfoy | Slytherin | 90
