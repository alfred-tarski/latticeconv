import torch
import os
from rivet import parse_rivet_output

x_bins = 40
y_bins = 40
categories = ['bathtub','bed','chair','desk','dresser','monitor','night','sofa','table','toilet']
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
Y.type(torch.LongTensor)
for index,file_name in enumerate(files):
    X[index,:,:,:] = parse_rivet_output(path+file_name,x_bins,y_bins)
    class_name = file_name.split('_')[1]
    v = label_dict[class_name]
    Y[index] = v
Y = Y.type(torch.LongTensor)

torch.save(X,'./rivet_features.pt')
torch.save(Y,'./rivet_classes.pt')