import torch.nn as nn

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