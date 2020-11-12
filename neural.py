import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import opt_einsum as oe

def build_meet_convolve_tensor(signal_dim,kernel_dim,kernel_loc):
  # produces a tensor M_ijk such that the contraction M_ijk f_i is equal to f_{j ^ k}
  M = torch.zeros(signal_dim,signal_dim,kernel_dim)
  # this is an inefficient implementation, but should be fast enough for the sizes we're doing.
  for i in range(signal_dim):
    for j in range(i,signal_dim):
      for k in range(kernel_dim):
        if i == min(j,k+kernel_loc):
          M[i,j,k] = 1
  return M

def build_join_convolve_tensor(signal_dim,kernel_dim,kernel_loc):
  # produces a tensor M_ijk such that the contraction M_ijk f_i is equal to f_{j v k}
  M = torch.zeros(signal_dim,signal_dim,kernel_dim)
  # this is an inefficient implementation, but should be fast enough for the sizes we're doing.
  for i in range(signal_dim):
    for j in range(i,signal_dim):
      for k in range(kernel_dim):
        if i == max(j,k+kernel_loc):
          M[i,j,k] = 1
  return M


# to compute a convolution (f*g)_{xy}, we need to calculate the contraction M_{ixa} N_{jyb} f_{ij} g_{ab}
def lattice_convolution_2d(convolve_x,convolve_y,signal,kernel):
  return oe.contract("ixa,jyb,ij,ab->xy",convolve_x,convolve_y,signal,kernel)
  #return torch.einsum("ixa,jyb,ij,ab->xy",convolve_x,convolve_y,signal,kernel)

class MeetConv2d(nn.Module):
  def __init__(self,signal_dim,kernel_dim,kernel_loc,in_features,out_features):
    super(MeetConv2d,self).__init__()
    (signal_x, signal_y) = signal_dim
    (kernel_x, kernel_y) = kernel_dim
    (loc_x, loc_y) = kernel_loc
    conv_x = build_meet_convolve_tensor(signal_x,kernel_x,loc_x)
    conv_y = build_meet_convolve_tensor(signal_y,kernel_y,loc_y)
    self.register_buffer('conv_x',conv_x)
    self.register_buffer('conv_y',conv_y)
    self.weights = nn.Parameter(torch.empty(kernel_x,kernel_y,in_features,out_features))
    self.bias = nn.Parameter(torch.empty(out_features,1,1))
    self.initialize_weights()

  def initialize_weights(self):
    nn.init.kaiming_normal_(self.weights)
    nn.init.zeros_(self.bias)
    
  def forward(self, X):
    Y = oe.contract("ixa,jyb,mfij,abfg->mgxy",self.conv_x,self.conv_y,X,self.weights)
    return Y + self.bias
  
  def extra_repr(self):
    kernel_x, kernel_y, in_features, out_features = self.weights.shape
    return 'meet convolution layer with input_features={}, output_features={}, kernel_size={}'.format(in_features, out_features, (kernel_x,kernel_y))

class JoinConv2d(nn.Module):
  def __init__(self,signal_dim,kernel_dim,kernel_loc,in_features,out_features):
    super(JoinConv2d,self).__init__()
    (signal_x, signal_y) = signal_dim
    (kernel_x, kernel_y) = kernel_dim
    (loc_x, loc_y) = kernel_loc
    conv_x = build_join_convolve_tensor(signal_x,kernel_x,loc_x)
    conv_y = build_join_convolve_tensor(signal_y,kernel_y,loc_y)
    self.register_buffer('conv_x',conv_x)
    self.register_buffer('conv_y',conv_y)
    self.weights = nn.Parameter(torch.empty(kernel_x,kernel_y,in_features,out_features))
    self.bias = nn.Parameter(torch.empty(out_features,1,1))
    self.initialize_weights()

  def initialize_weights(self):
    nn.init.kaiming_normal_(self.weights)
    nn.init.zeros_(self.bias)
    
  def forward(self, X):
    Y = oe.contract("ixa,jyb,mfij,abfg->mgxy",self.conv_x,self.conv_y,X,self.weights)
    return Y + self.bias
  
  def extra_repr(self):
    kernel_x, kernel_y, in_features, out_features = self.weights.shape
    return 'join convolution layer with input_features={}, output_features={}, kernel_size={}'.format(in_features, out_features, (kernel_x,kernel_y))

class LatticeCNN(nn.Module):
  def __init__(self,signal_dim,kernel_dim,n_features):
    super(LatticeCNN,self).__init__()
    self.meet_conv = []
    self.join_conv = []
    for i in range(len(n_features)-1):
      self.meet_conv.append(MeetConv2d(signal_dim,kernel_dim,(signal_dim[0]-kernel_dim[0],signal_dim[1]-kernel_dim[1]),n_features[i],n_features[i+1]).cuda())
      self.join_conv.append(JoinConv2d(signal_dim,kernel_dim,(0,0),n_features[i],n_features[i+1]).cuda())
    self.meet_conv = nn.ModuleList(self.meet_conv)
    self.join_conv = nn.ModuleList(self.join_conv)

  def forward(self,x):
    for (mc,jc) in zip(self.meet_conv,self.join_conv):
      x = F.leaky_relu((1-alpha)*mc(x) + alpha*jc(x))
      #x = F.max_pool2d(x,(2,2))
    return x

class LatticeCNNPool(nn.Module):
  def __init__(self,signal_dim,kernel_dim,n_features,pool_sizes,alpha=0.5):
    super(LatticeCNNPool,self).__init__()
    self.meet_conv = []
    self.join_conv = []
    dims = signal_dim
    self.pool_sizes = pool_sizes
    for i in range(len(n_features)-1):
      self.meet_conv.append(MeetConv2d(dims,kernel_dim,(dims[0]-kernel_dim[0],dims[1]-kernel_dim[1]),n_features[i],n_features[i+1]//2))
      self.join_conv.append(JoinConv2d(dims,kernel_dim,(0,0),n_features[i],n_features[i+1]//2))
      dims = (dims[0]//pool_sizes[i][0],dims[1]//pool_sizes[i][1])
    self.meet_conv = nn.ModuleList(self.meet_conv)
    self.join_conv = nn.ModuleList(self.join_conv)

  def forward(self,x,alpha=0.5):
    for (mc,jc,pool) in zip(self.meet_conv,self.join_conv,self.pool_sizes):
      #x = F.leaky_relu((1-alpha)*mc(x) + alpha*jc(x))
      x = F.leaky_relu(torch.cat((mc(x),jc(x)),dim=1))
      x = F.max_pool2d(x,pool)
    return x


class LatticeClassifier(nn.Module):
  def __init__(self,signal_dim,conv_layers,n_classes,p_drop=0.0):
    #e.g. conv_layers = [n_features,8,16,8,4]
    super(LatticeClassifier,self).__init__()
    self.convolutions = LatticeCNNPool(signal_dim,(4,4),[n_features,16,16,8],[(2,2),(2,2),(1,1)],alpha)
    self.convolutions.cuda()
    self.fc1 = nn.Linear(8*(signal_dim[0]//4)*(signal_dim[1]//4),32)
    self.fc2 = nn.Linear(32,32)
    self.fc3 = nn.Linear(32,n_classes)
    self.readout = nn.Linear(8*signal_dim[0]*signal_dim[1],n_classes)
    self.drop0 = nn.Dropout(p_drop)
    self.drop1 = nn.Dropout(p_drop)
    self.drop2 = nn.Dropout(p_drop)


  def forward(self,x):
    batch_size = x.shape[0]
    #x = self.drop0(x)
    x = self.convolutions(x)
    x = F.relu(self.drop1(self.fc1(torch.reshape(x,(batch_size,-1))))) #get rid of middle fc layer
    #x = self.fc1(torch.reshape(x,(batch_size, -1)))
    x = F.relu(self.drop2(self.fc2(x)))
    x = self.fc3(x)
    #x = torch.sum(x,dim=(2,3)) #dumbest possible classifier
    return x

class ConvClassifier(nn.Module):
  def __init__(self,signal_dim,n_features,n_classes, alpha=0.0, p_drop=0.0):
    super(ConvClassifier,self).__init__()
    self.convolutions = nn.ModuleList([nn.Conv2d(n_features,16,(4,4),1,padding=2),nn.Conv2d(16,16,(4,4),1,padding=2),nn.Conv2d(16,16,(4,4),1,padding=2)])
    self.fc1 = nn.Linear(16*(signal_dim[0])*(signal_dim[1]),32)
    self.fc2 = nn.Linear(32,32)
    self.fc3 = nn.Linear(32,n_classes)
    self.drop1 = nn.Dropout(p_drop)
    self.drop2 = nn.Dropout(p_drop)
  def forward(self,x):
    batch_size = x.shape[0]
    #for c in self.convolutions:
    #  x = F.leaky_relu(c(x))
    x = F.leaky_relu(self.convolutions[0](x)[:,:,0:40,0:40])
    x = F.max_pool2d(x,(2,2))
    x = F.leaky_relu(self.convolutions[1](x)[:,:,0:40,0:40])
    x = F.max_pool2d(x,(2,2))
    x = F.leaky_relu(self.convolutions[2](x)[:,:,0:40,0:40])
    #x = F.max_pool2d(x,(2,2))
    x = F.relu(self.drop1(self.fc1(torch.reshape(x,(batch_size,-1)))))
    #x = self.fc1(torch.reshape(x,(batch_size, -1)))
    x = F.relu(self.drop2(self.fc2(x)))
    x = self.fc3(x)
    #x = torch.sum(x,dim=(2,3)) #use the dumbest possible classifier
    return x

# class MLPClassifier(nn.Module):
#   def __init__(self,signal_dim,n_features,n_classes):
#     super(MLPClassifier,self).__init__()
#     self.fc1 = nn.Linear(n_features*(signal_dim[0])*(signal_dim[1]),32)
#     self.fc2 = nn.Linear(32,32)
#     self.fc3 = nn.Linear(32,n_classes)
#   def forward(self,x):
#     batch_size = x.shape[0]

#     x = F.relu(self.fc1(torch.reshape(x,(batch_size,-1))))
#     x = F.relu(self.fc2(x))
#     x = self.fc3(x)
#     return x

class HybridClassifier(nn.Module):
  def __init__(self,signal_dim,n_features,n_classes,alpha=0.5,p_drop=0.0):
    super(HybridClassifier,self).__init__()
    self.lattice_convolutions = LatticeCNNPool(signal_dim,(4,4),[n_features,8,8,8],[(2,2),(2,2),(2,2)],alpha)
    self.convolutions = nn.ModuleList([nn.Conv2d(n_features,8,(4,4),1,padding=2),nn.Conv2d(8,8,(4,4),1,padding=2),nn.Conv2d(8,8,(4,4),1,padding=2)])
    self.convolutions.cuda()
    self.fc1 = nn.Linear(16*(signal_dim[0]//8)*(signal_dim[1]//8),32)
    self.fc2 = nn.Linear(32,32)
    self.fc3 = nn.Linear(32,n_classes)
    self.drop1 = nn.Dropout(p_drop)
    self.drop2 = nn.Dropout(p_drop)

  def forward(self,x):
    batch_size = x.shape[0]
    x1 = self.lattice_convolutions(x)
    x2 = F.leaky_relu(self.convolutions[0](x)[:,:,0:40,0:40])
    x2 = F.max_pool2d(x2,(2,2))
    x2 = F.leaky_relu(self.convolutions[1](x2)[:,:,0:20,0:20])
    x2 = F.max_pool2d(x2,(2,2))
    x2 = F.leaky_relu(self.convolutions[2](x2)[:,:,0:10,0:10])
    x2 = F.max_pool2d(x2,(2,2))
    x = torch.cat((x1,x2),dim=1)
    x = F.relu(self.drop1(self.fc1(torch.reshape(x,(batch_size,-1))))) #get rid of middle fc layer
    #x = self.fc1(torch.reshape(x,(batch_size, -1)))
    x = F.relu(self.drop2(self.fc2(x)))
    x = self.fc3(x)
    #x = torch.sum(x,dim=(2,3)) #dumbest possible classifier
    
    return x