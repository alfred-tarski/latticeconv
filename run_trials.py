
import numpy as np
import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from rivet import parse_rivet_output
from neural import LatticeClassifier,ConvClassifier, HybridClassifier
from train import *
from numpy.random import permutation
from torch.utils.data import DataLoader,random_split
import time
from matplotlib.pyplot import figure,xlabel,ylabel,plot,savefig,legend,title
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

x_bins = 40
y_bins = 40

print("Loading dataset...")
categories = ['bathtub','bed','chair','desk','dresser','monitor','night','sofa','table','toilet']
n_features = 4
n_classes = len(categories)
feature_dim = (40,40)
label_dict = {category:index for index,category in enumerate(categories) }
reverse_label_dict = {index:category for index,category in enumerate(categories) }
X = torch.load('./rivet_features.pt')
Y = torch.load('./rivet_classes.pt')

print('data has shape: '+ str(X.shape))
print('labels has shape: ' + str(Y.shape))

## parameters for runs
n_trials = 4 
n_epochs = 100
alpha = 0
p_drop = 0.5
learning_rate = 5e-4
classifiers = [LatticeClassifier, ConvClassifier]#, HybridClassifier]
classifier_names = ["lattice","conv"]#,"hybrid"]
for (Classifier,name) in zip(classifiers,classifier_names):
    train_accuracy = torch.zeros(n_epochs,n_trials)
    test_accuracy = torch.zeros(n_epochs,n_trials)
    train_loss = torch.zeros(n_epochs,n_trials)
    #print("testing model '{:s}'".format(name))
    for trial in range(n_trials):
        trial_start = time.time()
        data = [[X[index,:,:,:],Y[index]] for index in range(X.shape[0])]
        training_data,testing_data = random_split(data,[len(data) - len(data)//10,len(data)//10],generator=torch.Generator().manual_seed(42+trial))
        trainloader = DataLoader(training_data,batch_size=128,shuffle=True,pin_memory=True)
        testloader = DataLoader(testing_data,batch_size=128,shuffle=False,pin_memory=True)
        print("{:s} trial {:d}".format(name, trial+1))
        model = Classifier(feature_dim,n_features,n_classes,alpha=alpha,p_drop=p_drop)
        model = model.to(device)
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        def callback(model,epoch):
            torch.save(model.state_dict(),"./checkpoints/{:s}_trial{:d}_epoch{:d}.pt".format(name,trial+1,epoch+1))
        train_accuracy[:,trial], test_accuracy[:,trial], train_loss[:,trial] = train(model, criterion, optimizer, trainloader, testloader, n_epochs, device, callback)
        print("Trial took {:.1f} seconds".format(time.time() - trial_start))
    torch.save(train_accuracy,'./{:s}_train_accuracy_t{:d}.pt'.format(name,n_trials))
    torch.save(test_accuracy,'./{:s}_test_accuracy_t{:d}.pt'.format(name,n_trials))
    torch.save(train_loss,'./{:s}_train_loss_t{:d}.pt'.format(name,n_trials))

#
