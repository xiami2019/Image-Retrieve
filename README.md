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
Triplet loss can estimate 

# Requirements
Python 3.6  
Pytorch 1.1.0  
TensorboardX  
 
# Network
This project use a pretrained Resnet18 to fine-tune on CUB_200_2011 and Stanford Dogs.
