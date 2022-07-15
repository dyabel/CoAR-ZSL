"""
Various helper network modules
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch.autograd as autograd
# from .CBAM import CBAMBlock
USE_CLASS_STANDARTIZATION = True # i.e. equation (9) from the paper
USE_PROPER_INIT = True # i.e. equation (10) from the paper



class MLP(nn.Module):
    """ a simple MLP"""

    def __init__(self, in_dim, sizes, out_dim, non_linearity):
        super().__init__()
        self.non_linearity = non_linearity
        self.in_layer = nn.Linear(in_dim, sizes[0])
        self.out_layer = nn.Linear(sizes[-1], out_dim)
        self.layers = nn.ModuleList([nn.Linear(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)])

    def forward(self, x):
        x = self.non_linearity(self.in_layer(x))
        for index, layer in enumerate(self.layers):
            if ((index % 2) == 0):
                x = self.non_linearity(layer(x))
        x = self.out_layer(x)
        return x


class ClassStandardization(nn.Module):
    """
    Class Standardization procedure from the paper.
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """

    def __init__(self, feat_dim: int):
        super().__init__()

        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad=False)

    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim=0)
            batch_var = class_feats.var(dim=0)

            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)

            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)

        return result

class ProtoModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int,with_cn: bool):
        super().__init__()
        self.fc1 = nn.Linear(attr_dim, hid_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim,hid_dim)
        if with_cn:
            self.cn1 = ClassStandardization(hid_dim)
            self.cn1_att = ClassStandardization(hid_dim)
          
        else:
            self.cn1 = nn.Identity()
            self.cn1_att = nn.Identity()
        self.relu2 = nn.ReLU()
        if with_cn:
            self.cn2 = ClassStandardization(hid_dim)
            self.cn2_att = ClassStandardization(hid_dim)
           
        else:
            self.cn2 = nn.Identity()
            self.cn2_att = nn.Identity()
        self.fc3 = nn.Linear(hid_dim,proto_dim)
        self.fc3_att = nn.Linear(hid_dim,proto_dim)
        self.relu3 = nn.ReLU()
        if USE_PROPER_INIT:
            weight_var = 1 / (hid_dim * proto_dim)
            b = np.sqrt(3 * weight_var)
            self.fc3.weight.data.uniform_(-b, b)
            self.fc3_att.weight.data.uniform_(-b, b)
          
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias,0)
        nn.init.constant_(self.fc2.bias,0)

       
    def forward(self, attrs, is_cls=True):
        if is_cls:
            protos = self.relu3(self.fc3(self.cn2(self.relu2(self.cn1(self.fc2(self.relu1(self.fc1(attrs))))))))
        else:
            protos = self.relu3(self.fc3_att(self.cn2_att(self.relu2(self.cn1_att(self.fc2(self.relu1(self.fc1(attrs))))))))
        return protos




def weight_init(*ms):
    for m in ms:
        if isinstance(m, torch.nn.Linear):
            size = m.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)

            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag())
    return torch.clamp(dist, 0.0, np.inf)

def cosine_similarity(x,y):
    x = torch.nn.functional.normalize(x,p=2,dim=1)
    y = torch.nn.functional.normalize(y,p=2,dim=1)
    # dist = 1. - x@y.T
    return x@y.T

def cosine_distance(x,y=None):
    x = torch.nn.functional.normalize(x,p=2,dim=1)
    if y is not None:
        y = torch.nn.functional.normalize(y,p=2,dim=1)
        dist = 1. - x @ y.T
    else:
        dist = 1. - x @ x.T
        dist = dist - torch.diag(dist.diag())
        dist = torch.clamp(dist,0,1)
    return dist
def euclid_distance(x,y):
    x = torch.nn.functional.normalize(x,p=2,dim=1)
    y = torch.nn.functional.normalize(y,p=2,dim=1)
    dist = 1. - x@y.T
    return dist

class CLASSIFIER(nn.Module):
    def __init__(self, input_dim,nclass,hid_dim=None):
        super(CLASSIFIER, self).__init__()
        self.hid_dim = hid_dim
        if hid_dim is not None:
            self.fc1 = nn.Linear(input_dim, hid_dim)
            self.fc2 = nn.Linear(hid_dim, nclass)
        else:
            self.fc1 = nn.Linear(input_dim, nclass)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.constant(self.fc.bias,0)
        # self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x):
        if self.hid_dim is not None:
            o = self.fc2(F.relu(self.fc1(x)))
        else:
            o = self.fc1(x)
        return o



