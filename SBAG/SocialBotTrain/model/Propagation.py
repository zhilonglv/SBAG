 #from collections import OrderedDict
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain



# 模型架构
class BD(nn.Module):
    def __init__(self, in_dim, mem_dim, Nclass):
        super(BD, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.linear = nn.Linear(in_dim, mem_dim)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, user):
        
        output = self.linear(user.unsqueeze(1))#[hi],ht#[5,1,16] [batch,1,emd]
        output = output.squeeze(1)#[5,16]
        output = torch.tanh(output)

        return output

if __name__ == '__main__':
    x = torch.Tensor([[17, 49, 13, 4.07753744390572, 2.5649493574615367, 0.0, 0.0, 8.48920515487607, 93, 1, 0, 0, 0, 1, 1, 0]])#[1,16]
    model = BD(16,16,2)
    output = model(x)
    print(output)