import torch.nn as nn
import torch
import numpy as np
from numpy.polynomial.hermite import hermgauss
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import string

class ae_dist():

    def __init__(self,mu,L,d):

        self.data = mu
    
    def rsample(self):

        return self.data 

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
    
    def posterior(self,target,n_quadrature_points,log_prob_fnc):


        # Generate 1D Gauss-Hermite nodes and weights
        x_1d, w_1d = hermgauss(n_quadrature_points)
        
        # Adjust nodes and weights for standard normal distribution
        nodes_1d = x_1d * np.sqrt(2)          # Scale nodes
        weights_1d = w_1d / np.sqrt(np.pi)    # Scale weights
        
        # Create 2D grid via tensor product
        x_grid, y_grid = np.meshgrid(nodes_1d, nodes_1d, indexing='ij')
        weights_2d = np.outer(weights_1d, weights_1d)
        log_prior = torch.from_numpy(weights_2d.flatten()).to(self.device) ## these are approoooximately log p(z), (2500,)
        xy_grid = torch.from_numpy(np.stack([x_grid.flatten(),y_grid.flatten()],axis=1)).to(torch.float32).to(self.device) #(2500,2)
        log_likelihood = log_prob_fnc(self.decoder(xy_grid),target) ## these are approooooooxxxxximately log p(x|z) (2500,1)
        (mu,L,d) = self.encoder(target)
        dist = LowRankMultivariateNormal(mu,L,d)
        lp_grid_encoder = dist.log_prob(xy_grid) #(2500,)

        # Evaluate f on grid and compute integral
        log_evidence = torch.special.logsumexp(log_likelihood + log_prior[:,None],dim=0) # this is approoooooooooooxxximately log p(x) #(1,)
        log_posterior = log_likelihood.squeeze() + log_prior.squeeze() - log_evidence.squeeze() # (2500,)
        return nn.Softmax(dim=0)(log_posterior).detach().cpu().numpy().squeeze(),nn.Softmax(dim=0)(lp_grid_encoder).detach().cpu().numpy(),xy_grid.detach().cpu().numpy()
    
    def embed_data(self,loader):
        latents = []
        labels = [] 
        with torch.no_grad():
            for (data,label) in loader:
                data = data.to(self.device).to(torch.float32)
                if type(label) == tuple:
                    labels.append([string.ascii_lowercase.index(l.lower()[0]) for l in label])
                else:
                    labels.append(label.detach().cpu().numpy())

                lat,_,_ = self.encode(data)
                # posterior is B x S, convert to B x 2 for weighted grid
                latents.append(lat)

        latents = torch.vstack(latents).detach().cpu().numpy()
        labels = np.hstack(labels)
        return latents,labels
    

class IWAE(VAE):

    def __init__(self,decoder,encoder,distribution,device,k_samples=10):


        super(IWAE,self).__init__(decoder,encoder,distribution,device)

        self.k_samples=k_samples

    def forward(self,x):

        x = x.to(torch.float32)
        params = self.encoder(x)

        dist = self.distribution(*params)
        z = dist.rsample([self.k_samples]) # K x B x d 
        recons = torch.vmap(self.decoder,in_dims=(1),out_dims=(1))(z).permute(1,0,2,3,4) # KxBxCxHxW->BxKxCxHxW

        return recons,(z,dist)
    
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