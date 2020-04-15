# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:02:47 2020

@author: hp1
"""
###Get the data from 
#https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

import os
import cv2
import numpy as np
from tqdm import tqdm #progress bar
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#########################Preprocessing########################################
REBULID_DATA = False #So we wont preprocess the data again and again

class DogsVSCats():
    IMG_SIZE = 50 #resizing the imgs to 50x50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS:1}
    training_data = []
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    #try np.eye(5) it's like matlab creating ones matrix
                    #we will convert this labels to one hot or vector class
                    #so instead of 0 for cat and 1 for dogs (scalar)
                    #it will be [1, 0] for cats and [0, 1] for dogs
                    
                    #Labelling the data. Adding the img and label to a list
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    
                    if label == self.CATS:
                        self.catcount +=1
                    elif label == self.DOGS:
                        self.dogcount +=1
                except Exception as e:
                    print('================================')
                    print(str(e))
                    pass
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)

if REBULID_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()        
#####################################Now building the model#############################################                

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ################Defining convolusion layers#####################
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1, output is 32 convulusion features
                                        # 5 is 5x5 kernal
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        #######Now converting these 2d layers in a 1-d (linear) layer################
        ########it's hard##############
        ##########For linear layers, we need to find the input size and for that
        ##########We will create a random data and pass it through the conv layers
        ##########Then we will check the len of the output and that will be the
        #######input for the linear layer
        x = torch.randn(50,50).view(-1,1,50,50) #randn means can go to negative
                                                #(1,50,50) is the image size
                                                # -1 all the number of featuresets from conv layers
        self._to_linear = None #defining input variable for linear layer
        self.convs(x)  #########calling the forward function for conv layers defined below
        ####################################
        ###########Now difining the linear layer###########################
        self.fc1 = nn.Linear(self._to_linear, 512) #inital layer
        self.fc2 = nn.Linear(512, 2) #Output layer. 2 because we have 2 classes
        
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) #Applying the pooling matrix after the activation
                                                        #(2,2) is the size of pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]  #Thats the size to be used for the input layer                                        
        return x
    def forward (self, x): #Now defining the real direction
        x = self.convs(x)  #calling the real direction for conv layer
        x = x.view(-1, self._to_linear) #making the dimension of the output same as the expected input for the linear layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)

###########################################################################
training_data = np.load("training_data.npy", allow_pickle=True)

plt.imshow(training_data[1][0], cmap=  'gray') #showing an img
plt.show()
####################################calling the model####################
net=Net()
print(net)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss() #here the loss function is mean square error

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0 #scaling to 0-1
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1 # dividing the data
val_size = int(len(X)*VAL_PCT) 

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

BATCH_SIZE = 100
EPOCHS = 3
print('=============================================================')
print(input("Press Enter"))
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_x = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]
        net.zero_grad()
        outputs = net(batch_x)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(loss)
    
correct=0
total=0

with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct +=1
        total +=1
print("Accuracy: ", correct/total)
        