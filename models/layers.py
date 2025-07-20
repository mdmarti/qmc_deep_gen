import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.distributed as dist
#from third_party.inplace_sync_batchnorm import SyncBatchNormSwish


class LinearResBlock(nn.Module):

    def __init__(self,in_size,hidden_size=125):

        super(LinearResBlock,self).__init__()

        self.hidden = nn.Sequential(nn.BatchNorm1d(num_features=in_size),
                                    nn.Linear(in_size,hidden_size),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(num_features=hidden_size),
                                    nn.Linear(hidden_size,in_size))

    def forward(self,x):

        return x + self.hidden(x)

class ConvResBlock(nn.Module):

    def __init__(self,in_channels,hidden_channels=32,kernel_size=3,pad=1):

        super(ConvResBlock,self).__init__()

        self.hidden = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),
                                    nn.Conv2d(in_channels,hidden_channels,kernel_size,padding=pad),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=hidden_channels),
                                    nn.Conv2d(hidden_channels,in_channels,kernel_size,padding=pad))

    def forward(self,x):

        return x + self.hidden(x)
    
class MLPResCellNVAESimple(nn.Module):

    def __init__(self,in_size,expand_factor=4):

        super(ResCellNVAESimple,self).__init__()
        #self.ops = nn.ModuleList()

        op1 = nn.BatchNorm1d(num_features=in_size)
        op2 = nn.Linear(in_size,expand_factor*in_size)
        op3_1 = nn.BatchNorm1d(num_features=expand_factor*in_size)
        op3_2 = nn.SiLU()
        op4 = nn.Linear(expand_factor*in_size,expand_factor*in_size)
        op5_1 = nn.BatchNorm1d(num_features=expand_factor*in_size)
        op5_2 = nn.SiLU()
        op6= nn.Linear(expand_factor*in_size,in_size,1)
        op7 = nn.BatchNorm1d(num_features=in_size)
        op8 = SE(in_size,in_size)

        self.ops = nn.Sequential(op1,op2,op3_1,op3_2,op4,op5_1,op5_2,op6,op7,op8)

    def forward(self,x):

        return x + 0.1*self.ops(x)

class ResCellNVAESimple(nn.Module):

    def __init__(self,in_size,expand_factor=4):

        super(ResCellNVAESimple,self).__init__()
        #self.ops = nn.ModuleList()

        op1 = nn.BatchNorm2d(num_features=in_size)
        op2 = nn.Conv2d(in_size,expand_factor*in_size,1)
        op3_1 = nn.BatchNorm2d(num_features=expand_factor*in_size)
        op3_2 = nn.SiLU()
        op4 = nn.Conv2d(expand_factor*in_size,expand_factor*in_size,5,groups=expand_factor*in_size,padding=2)
        op5_1 = nn.BatchNorm2d(num_features=expand_factor*in_size)
        op5_2 = nn.SiLU()
        op6= nn.Conv2d(expand_factor*in_size,in_size,1)
        op7 = nn.BatchNorm2d(num_features=in_size)
        op8 = SE(in_size,in_size)

        self.ops = nn.Sequential(op1,op2,op3_1,op3_2,op4,op5_1,op5_2,op6,op7,op8)

    def forward(self,x):

        return x + 0.1*self.ops(x)
    
#### from https://github.com/NVlabs/NVAE/tree/master
class SE(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se
    
class ZeroLayer(nn.Module):
    def __init__(self):

        super(ZeroLayer,self).__init__()

    def forward(self,x):

        return 0 * x
    
class PermutationLayer(nn.Module):

    def __init__(self,permute_type='output'):
        super(PermutationLayer,self).__init__()

        self.permute_type = permute_type
        self.p_order = [0,2,3,1] if permute_type =='output' else [0,3,2,1]

    def forward(self,x):
        # permutes dimensions to be BxHxWxC, from BxCxHxW
        # if dims are BxHxWxC, needs to be  (0,3,1,2)
        return x.permute(*self.p_order)