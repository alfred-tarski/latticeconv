import numpy as np
import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from rivet import parse_rivet_output
from neural import LatticeClassifier,ConvClassifier
from train import *
from numpy.random import permutation
from torch.utils.data import DataLoader,random_split
import time
from matplotlib.pyplot import figure,xlabel,ylabel,plot,savefig,legend,title
import pickle

#torch.manual_seed(11)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

x_bins = 40
y_bins = 40

<<<<<<< Updated upstream
#binary classificaiton
=======
print("Loading dataset...")
#binary classification
>>>>>>> Stashed changes
categories = ['sofa','monitor']
n_features = 4
n_classes = len(categories)
feature_dim = (40,40)
label_dict = {category:index for index,category in enumerate(categories) }
reverse_label_dict = {index:category for index,category in enumerate(categories) }
path = './invariants/'
N = len(os.listdir(path))
files = os.listdir(path)
X = torch.zeros(N,n_features,x_bins,y_bins)
Y = torch.zeros(N)
for index,file_name in enumerate(files):
    X[index,:,:,:] = parse_rivet_output(path+file_name,x_bins,y_bins)
    v = label_dict[file_name.split('_')[1]]
    Y[index] = v
Y = Y.type(torch.LongTensor)

print('data has shape: '+ str(X.shape))
print('labels has shape: ' + str(Y.shape))

#binary classification sofa vs. monitor
n_trials = 10
n_epochs = 50
p_drop = 0.5
train_accuracy = torch.zeros(n_epochs,n_trials)
test_accuracy = torch.zeros(n_epochs,n_trials)
#ConvClassifier
for trial in range(n_trials):
    data = [[X[index,:,:,:],Y[index]] for index in range(X.shape[0])]
    training_data,testing_data = random_split(data,[len(data) - len(data)//10,len(data)//10],generator=torch.Generator().manual_seed(42+trial))
    trainloader = DataLoader(training_data,batch_size=16,shuffle=True,pin_memory=True)
    testloader = DataLoader(testing_data,batch_size=16,shuffle=False,pin_memory=True)
    print("Trial {:d}".format(trial+1))
    model = ConvClassifier(feature_dim,n_features,n_classes,p_drop)
    model = model.to(device)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    def callback(model,epoch):
        torch.save(model.state_dict(),"./checkpoints/lattice_trial{:d}_epoch{:d}".format(trial+1,epoch+1))
    train_accuracy[:,trial], test_accuracy[:,trial], _ = train(model, criterion, optimizer, trainloader, testloader, n_epochs, device, callback)
    print("Trial took {:.1f} seconds".format(time.time() - trial_start))
torch.save(train_accuracy,'./conv_train_accuracy.pt')
torch.save(test_accuracy,'./conv_test_accuracy.pt')
