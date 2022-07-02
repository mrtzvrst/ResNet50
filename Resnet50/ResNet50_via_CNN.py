import numpy as np
import math
import h5py
import torch
from torch import nn, optim
from cnn_utils import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

"""
Run the following cell to train your model on 2 epochs with a batch size of 32. 
On a CPU it should take you around 5min per epoch.
"""

def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


"""Conv resnet"""
class Convolution_BLCK(nn.Module):
    def __init__(self, n_ch, F1, F2, F3, f, s):
        super(Convolution_BLCK, self).__init__()
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(n_ch, F1, (1,1), stride=(s,s), padding='valid'),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.Conv2d(F1, F2, (f,f), stride=(1,1), padding='same'),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(F2, F3, (1,1), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(F3))
        self.ShortCut = nn.Sequential(
            nn.Conv2d(n_ch, F3, (1,1), stride=(s,s), padding='valid'),
            nn.BatchNorm2d(F3))
        self.relu = nn.Sequential(
            nn.ReLU())
    def forward(self, x):
        return self.relu(self.ConvBlock(x) + self.ShortCut(x))
    
"""Identity resnet"""
class Identity_BLCK(nn.Module):
    def __init__(self, n_ch, F1, F2, F3, f, times):
        super(Identity_BLCK, self).__init__()
        self.time = times
        self.IdentBlock = nn.Sequential(
            nn.Conv2d(n_ch, F1, (1,1), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.Conv2d(F1, F2, (f,f), stride=(1,1), padding='same'),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(F2, F3, (1,1), stride=(1,1), padding='valid'),
            nn.BatchNorm2d(F3))
        self.relu = nn.Sequential(
            nn.ReLU())
    def forward(self, x):
        for i in range(self.time):
            x = self.relu(self.IdentBlock(x) + x)
        return x
        

class ResNet50(nn.Module):
    def __init__(self, Inp_ch, Out_ch):
        super(ResNet50, self).__init__()
        self.out_ch = Out_ch
        """First stage"""
        self.stage1 = nn.Sequential(
            nn.Conv2d(Inp_ch, 64, (7,7), stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3,3), stride=(2,2))
            )
        """stage 2"""
        self.stage2 = nn.Sequential(
            Convolution_BLCK(64, 64, 64, 256, 3, 1),
            Identity_BLCK(256, 64, 64, 256, 3, 2)
            )
        """stage 3"""
        self.stage3 = nn.Sequential(
            Convolution_BLCK(256, 128, 128, 512, 3, 2),
            Identity_BLCK(512, 128, 128, 512, 3, 3)
            )
        """stage 4"""
        self.stage4 = nn.Sequential(
            Convolution_BLCK(512, 256, 256, 1024, 3, 2),
            Identity_BLCK(1024, 256, 256, 1024, 3, 5)
            )
        """stage 5"""
        self.stage5 = nn.Sequential(
            Convolution_BLCK(1024, 512, 512, 2048, 3, 2),
            Identity_BLCK(2048, 512, 512, 2048, 3, 2)
            )
        
        self.FC = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(8192, Out_ch)
            )
            
    def forward(self, x):
        x = torch.nn.functional.pad(x, (3, 3, 3, 3))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.FC(x)
            
# X_tr = torch.tensor(X_train)
# m = ResNet50(3, 6)
# print(m(X_tr[0:1]).shape)


def model(X_train, Y_train, X_test, Y_test, classes, learning_rate = 0.001,
          num_epochs = 1500, minibatch_size = 32, Lambda = 0.000001, print_cost = True):
    
    (m, c, h, w) = X_train.shape
    n_y = len(classes)  
    
    costs = []
    
    model = ResNet50(c, n_y)
    CEF_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    seed = 0
    for epoch in range(num_epochs):
        epoch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        
        for minibatch_X, minibatch_Y in minibatches:
            optimizer.zero_grad()
            Y_out = model.forward(minibatch_X)
            loss = CEF_loss(Y_out,minibatch_Y) #+ Lambda* sum([torch.sum(p**2) for p in model.parameters()])
            loss.backward()
            optimizer.step()
            
            epoch_cost += loss.item() / num_minibatches
        if print_cost == True and epoch % 1 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 10 == 0:
            costs.append(epoch_cost)
    
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
   
    return model


# Loading the data (signs)
X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()

# Example of a picture
index = 3
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.


(m, h, w, c) = X_train.shape
X_train= X_train.reshape((m, c, h, w))
(m, h, w, c) = X_test.shape
X_test = X_test.reshape((m, c, h, w))
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
"""Training the model"""
trained_model = model(torch.tensor(X_train), torch.LongTensor(Y_train.flatten()), 
                      torch.tensor(X_test), torch.LongTensor(Y_test.flatten()), 
                      classes,  learning_rate = 0.001,  num_epochs = 100, minibatch_size = 32)

"""Prediction"""
trained_model.eval()
Y_out = torch.argmax(trained_model(torch.tensor(X_test)), dim=1)
accuracy = (1-torch.sum(Y_out!=torch.LongTensor(Y_test.flatten()))/Y_out.shape[0])*100
print('accuracy is: %f'%accuracy)





















