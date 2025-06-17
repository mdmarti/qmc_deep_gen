import torch.nn as nn
import torch

class VAE(nn.Module):

    def __init__(self,decoder,encoder,distribution,device):


        super(VAE,self).__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.distribution = distribution
        self.device=device

        self.to(device)

    def forward(self,x):
        x = x.to(torch.float32)
        params = self.encoder(x)

        dist = self.distribution(*params)
        z = dist.rsample()
        recons = self.decoder(z)

        return recons,params
    
    def encode(self,x):

        return self.encoder(x)
    
    def decode(self,z):

        return self.decoder(z)
    
    def round_trip(self,x):

        (mu,_,_) = self.encoder(x)
        return self.decoder(mu)


    
class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x
        
class Encoder(nn.Module):


    def __init__(self,net=None,mu_net=None,l_net=None,d_net=None,latent_dim=2):


        super(Encoder,self).__init__()
        self.latent_dim=latent_dim

        if net is None:
            #assuming a 1x128x128 input,
            self.shared_net = nn.Sequential(
                nn.Conv2d(1,8,3,stride=2,padding=1), #B x  8 x 64 x 64
                nn.ReLU(),
                #Print(),
                nn.Conv2d(8,16,3,stride=2,padding=1), #B x 16 x 32 x 32
                nn.ReLU(),
                #Print(),
                nn.Conv2d(16,32,3,stride=2,padding=1), #B x 32 x 16 x 16
                nn.ReLU(),
                #Print(),
                nn.Conv2d(32,64,3,stride=2,padding=1), #B x 64 x 8 x 8,
                nn.ReLU(),
                #Print(),
                nn.Flatten(start_dim=1,end_dim=-1), # B x 8*8*64,
                #Print(),
                nn.Linear(8*8*64,2**11),
                nn.ReLU(),
                #Print(),
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
        #print(data.shape)
        intermediate = self.shared_net(data)
        #assert False
        mu = self.mu_net(intermediate)
        l = self.mu_net(intermediate).unsqueeze(-1)
        d = self.mu_net(intermediate).exp()

        return (mu,l,d)