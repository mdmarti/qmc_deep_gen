import torch.nn as nn


class VAE(nn.Module):

    def __init__(self,decoder,encoder,distribution,device):


        super(VAE,self).__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.distribution = distribution

        self.to(device)

    def forward(self,data):

        params = self.encoder(data)

        dist = self.distribution(*params)
        z = dist.rsample()
        recons = self.encoder(z)

        return params,recons
    
    def encode(self,data):

        return self.encoder(data)
    
    def decode(self,z):

        return self.decoder(z)
    

class Encoder(nn.module):


    def __init__(self):


        super(Encoder,self).__init__()

        