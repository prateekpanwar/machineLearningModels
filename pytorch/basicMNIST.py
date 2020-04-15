# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:28:48 2020

@author: hp1
"""

import torch
import torch.nn as nn #Oops structures
import torch.nn.functional as F #Most the functions
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.optim as optim #Optimizer 

#get OCR data
train = datasets.MNIST("", train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
#transforming for the data to fit

test = datasets.MNIST("", train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))

#Now creating 2 more variables for iterrating over the train and test
#because in deep learning, you only take a chunk of data at a time
#Coz in case of billion data, the memorysize can be full before whole data can be fit
# Always use the size between 8-64
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# understanding the flow and visualizing
for data in trainset:
    print(data)
    break
x,y = data[0][0], data[1][0] #last batch 0th element and try to reconstruct the img
print(x.shape) # checking if the data is in the form of an image or no. It's not
                #We are looking for a 2-d image and the array is 3d

plt.imshow(x.view(28,28))

#Now checking and Balancing each class (here 0-9)
total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] +=1
        total = total+1
print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100}")

###########################Writing the model##############################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ###############Defining layers################
        self.fc1 = nn.Linear(28*28, 64) #fc1 means fully connected layer 1. Proving input (img size) and output (number of neurons)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) #output layer
        
    #Forward function. Defining the path of the data through the layers
    def forward(self, x):
        x = F.relu(self.fc1(x)) #F.relu is the activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) #No need of activation on the output layer
        return F.log_softmax(x, dim=1) #return the probability distribution. dim=1 will always be the same


net = Net()
print(net)

###################testing the model with random variable#####################

X = torch.rand((28,28))
X = X.view(-1, 28*28) #Reshaping the array. Here -1 means unknown size
output = net(X)
print(output)
    
########################Optimizer and loss function######################################

optimizer = optim.Adam(net.parameters(), lr=0.001) #Here adam is commonly used optimizer. net.parameter is what are
                                                    #the parameter that can be optimized and lr is learning rate
                                                    #Adjusting the weights based on the error(loss)
####################Training#########################################################
EPOCHS = 3 # 3 times passing the whole data for training                                                    

for epoch in range(EPOCHS): # 3 full passes over the data
    for data in trainset:  # data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                        # contains loss values that optimizer uses
        output = net(X.view(-1,28*28))  # pass in the reshaped batch (recall they are 28x28 atm)
        loss = F.nll_loss(output, y)  # calc and grab the loss value
                                        #nll_loss is used when the classes are scaler
                                        #or else we use mean square error
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 

###############Now see how correct were we (accuracy)###################################

correct = 0
total = 0

#comparing the predicted label and the real labels
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct = correct+1
            total = total+1
print("Accuracy: ", correct/total)
#####################################Visualizing the accuracy#####################

plt.imshow(X[5].view(28,28))
plt.show()
print(torch.argmax(net(X[5].view(-1, 28*28))[0]))