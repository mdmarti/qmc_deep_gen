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


    def __init__(self,net=None,mu_net=None,l_net=None,d_net=None,latent_dim=2):


        super(Encoder,self).__init__()
        self.latent_dim=latent_dim

        if net is None:
            #assuming a 1x128x128 input,
            self.shared_net = nn.Sequential(
                nn.Conv2d(1,8,3,stride=2,padding=1), #B x  8 x 64 x 64
                nn.ReLU(),
                nn.Conv2d(8,16,3,stride=2,padding=1), #B x 16 x 32 x 32
                nn.ReLU(),
                nn.Conv2d(16,32,3,stride=2,padding=1), #B x 32 x 16 x 16
                nn.ReLU(),
                nn.Conv2d(32,64,3,stride=2,padding=1), #B x 8 x 64 x 64,
                nn.ReLU(),
                nn.Flatten(start_dim=1,end_dim=-1), # B x 8*64*64,
                nn.Linear(8*64*64,2**11),
                nn.ReLU()
            )
            self.mu_net = nn.Linear(2**11,latent_dim)
            self.l_net = nn.Linear(2**11,latent_dim)
            self.d_net = nn.Linear(2**11,latent_dim) 
        else:
            self.shared_net = net
            self.mu_net = mu_net
            self.l_net = l_net
            self.d_net = d_net


    def forward(self,data):

        intermediate = self.shared_net(data)

        mu = self.mu_net(intermediate)
        l = self.mu_net(intermediate).unsqueeze(-1)
        d = self.mu_net(intermediate).exp()

        return (mu,l,d)