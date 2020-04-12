# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:33:44 2020

@author: hp1
"""

import torch

x = torch.Tensor([5,3]) #Tenson is just a multi dimensional array. Easily be used in GPU
y = torch.Tensor([2,1])

print(x*y)


x = torch.zeros([2,5])
print(x)
print(x.shape) #get size of array

y = torch.rand([2,5])
print(y)

print(y.view([1,10])) #reshaping in pytorch

