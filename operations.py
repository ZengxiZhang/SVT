import os
import sys
import time 
import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn
# import tflearn
# import tf_slim as slim

# OPS = {
#   'none' : lambda x, C, stride: Zero(x,stride),
#   'avg_pool_3x3' : lambda x, C, stride: slim.avg_pool2d(x,[3,3], stride=stride,padding='SAME'),
#   'max_pool_3x3' : lambda x, C, stride: slim.max_pool2d(x,[3,3], stride=stride,padding='SAME'),
#   'skip_connect' : lambda x, C, stride: tf.identity(x) if stride == [1,1] else FactorizedReduce(x, C),
#   'sep_conv_3x3' : lambda x, C, stride: SepConv(x, C, [3,3], stride),
#   'sep_conv_5x5' : lambda x, C, stride: SepConv(x, C, [5,5], stride),
#   'sep_conv_7x7' : lambda x, C, stride: SepConv(x, C, [7,7], stride),
#   'dil_conv_3x3' : lambda x, C, stride: DilConv(x, C, [3,3], stride,2),
#   'dil_conv_5x5' : lambda x, C, stride: DilConv(x, C, [5,5], stride,2),
#   # 'conv_7x1_1x7' : lambda x, C, stride: nn.Sequential(
#   #   nn.ReLU(inplace=False),
#   #   nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
#   #   nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
#   #   nn.BatchNorm2d(C, affine=affine)
#   #   ),
# }

        
OPS = {
  'conv_1x1' : lambda C_in, C_out, stride, batchnorm=True: Conv(C_in, C_out, 1, stride,batchnorm),
  'conv_3x3' : lambda C_in, C_out, stride, batchnorm=True: Conv(C_in, C_out, 3, stride,batchnorm),
  'conv_5x5' : lambda C_in, C_out, stride, batchnorm=True: Conv(C_in, C_out, 5, stride,batchnorm),
  'dil_conv_3x3' : lambda C_in, C_out, stride, batchnorm=True: DilConv(C_in, C_out, 3, stride,2,batchnorm),
  'dil_conv_5x5' : lambda C_in, C_out, stride, batchnorm=True: DilConv(C_in, C_out, 5, stride,2,batchnorm)
  }

# def Zero(x,stride):
# 	return tf.zeros_like(x)[:,::stride[0],::stride[1],:]

class DilConv(nn.Module):
	def __init__(self, C_in, C_out, kernel_size, stride, rate,batchnorm):
		super(DilConv, self).__init__()
		pad=(kernel_size//2)*2
		# print(kernel_size)
		# print(pad)
		if batchnorm:
			self.op = torch.nn.Sequential( nn.Conv2d(in_channels=C_in, out_channels=C_out, stride=stride,kernel_size=kernel_size, padding=pad,dilation=rate),
												nn.BatchNorm2d(C_out),
												nn.ReLU(True))
		else:
			self.op = torch.nn.Sequential( nn.Conv2d(in_channels=C_in, out_channels=C_out, stride=stride,kernel_size=kernel_size, padding=pad,dilation=rate),
												nn.ReLU(True))
	def forward(self, x):
		# print(x.shape)
		x=self.op(x)
		# print(x.shape)
		return x

class Conv(nn.Module):
	def __init__(self, C_in, C_out, kernel_size, stride,batchnorm):
		super(Conv, self).__init__()
		pad = kernel_size//2

		if batchnorm:
			self.op = torch.nn.Sequential( nn.Conv2d(in_channels=C_in, out_channels=C_out, stride=stride,kernel_size=kernel_size, padding=pad),
											nn.BatchNorm2d(C_out),
											nn.ReLU(True))
		else:
			self.op = torch.nn.Sequential( nn.Conv2d(in_channels=C_in, out_channels=C_out, stride=stride,kernel_size=kernel_size, padding=pad),
											nn.ReLU(True))
	def forward(self, x):
		# print(x.shape)
		x=self.op(x)
		# print(x.shape)
		return x
	# # time.sleep(2)
	# if batchnorm:
	# 	x=tflearn.relu(x)
	# 	# C_in=x.get_shape()[-1].value
	# 	x = slim.separable_convolution2d(x, C_out, kernel_size, depth_multiplier = 1, stride = stride)
	# 	x = slim.batch_norm(x)
	# else:
	# 	x = slim.conv2d(inputs = x, num_outputs = C_out ,kernel_size = kernel_size, activation_fn = tf.nn.relu)


	# x=slim.separable_convolution2d(x,C_out,kernel_size,depth_multiplier=1)
	# x=slim.batch_norm(x)
	# return x
# def ResConv(x,c_out,kernel_size,stride):
# 	x=tflearn.relu(x)
# 	# C_in=x.get_shape()[-1].value
# 	C_in=x.shape[-1]
# 	x_out=slim.separable_convolution2d(x,c_out,kernel_size,depth_multiplier=1,stride=stride)
# 	x_out=slim.batch_norm(x_out)
# 	# x=tf.concat((x,x_out),-1)
# 	if stride==[2,2]:
# 		x=slim.max_pool2d(inputs=x, kernel_size=2, padding='SAME')
# 	x+=x_out
# 	return x

# def ResDilConv(x,C_out,kernel_size,stride,rate):
# 	x=tflearn.relu(x)
# 	C_in=x.shape[-1]
# 	x_out=slim.separable_convolution2d(x,C_out,kernel_size,depth_multiplier=1,stride=stride,rate=rate)
# 	x_out=slim.batch_norm(x_out)
# 	if stride==[2,2]:
# 		x=slim.max_pool2d(inputs=x, kernel_size=2, padding='SAME')
# 	# x=tf.concat((x,x_out),-1)
# 	x+=x_out
# 	return x
# def FactorizedReduce(x,c_out):
# 	# print(x.shape)
# 	# print(c_out//2)
	
# 	x=tflearn.relu(x)
# 	conv1=slim.conv2d(x,c_out//2,[1,1],stride=[2,2])
# 	# print(conv1.shape)
# 	conv2=slim.conv2d(x[:,1:,1:,:],c_out//2,[1,1],stride=[2,2])
# 	# print(conv2.shape)
# 	x=tf.concat([conv1,conv2],-1)
# 	# print(x.shape)
# 	x=slim.batch_norm(x)
# 	# time.sleep(1000)
# 	return x
# def ReLUConvBN(x,C_out):
# 	x=tflearn.relu(x)
# 	x=slim.conv2d(x,C_out,[3,3])
# 	x=slim.batch_norm(x)
# 	return x