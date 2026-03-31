import torch
from torch import nn 
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import numpy as np
from tqdm import tqdm

class VAE(nn.Module):

    def __init__(self,decoder,encoder,distribution,device):
        """
        Takes as input:
            decoder (nn.Module): Decoder network. Should output a set of parameters for
            whatever latent distribution this VAE will have over the latent space
            encoder (nn.Module): Encoder network. maps from latent space to data space

            distribution (torch.distributions.Distribution): Distribution over the latent space. 
            Used for reparameterized sampling from the encoder distribution

            device: torch.device('cuda') or torch.device('cpu') 
        """


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
    

    def embed_data(self,loader):
        latents = []
        labels = [] 
        with torch.no_grad():
            for (data,label) in tqdm(loader,desc='embedding latents',total=len(loader)):
                data = data.to(self.device).to(torch.float32)

                labels.append(label.detach().cpu().numpy())

                lat,_,_ = self.encode(data)
                # posterior is B x S, convert to B x 2 for weighted grid
                latents.append(lat)

        latents = torch.vstack(latents).detach().cpu().numpy()
        labels = np.hstack(labels)
        return latents,labels
    
class IWAE(VAE):

    def __init__(self,decoder,encoder,distribution,device,k_samples=10):
        """
        Importance weighted AE. Requires same inputs as VAE, with an additional
        k_samples (int): Number monte carlo samples from encoder distribution
        """


        super(IWAE,self).__init__(decoder,encoder,distribution,device)

        self.k_samples=k_samples

    def forward(self,x):

        x = x.to(torch.float32)
        params = self.encoder(x)

        dist = self.distribution(*params)
        z = dist.rsample([self.k_samples]) # K x B x d 
        recons = torch.vmap(self.decoder,in_dims=(1),out_dims=(1))(z).permute(1,0,2,3,4) # KxBxCxHxW->BxKxCxHxW

        return recons,(z,dist)
    

class Encoder(nn.Module):

    def __init__(self,net,mu_net,l_net,d_net,latent_dim=2):


        super(Encoder,self).__init__()
        self.latent_dim=latent_dim

        self.shared_net = net
        self.mu_net = mu_net
        self.l_net = l_net
        self.d_net = d_net

    def forward(self,data):
        #print(data.shape)
        intermediate = self.shared_net(data)
        #assert False
        mu = self.mu_net(intermediate)
        l = self.l_net(intermediate).unsqueeze(-1)
        d = self.d_net(intermediate).exp()

        return (mu,l,d)
    
class ZeroLayer(nn.Module):
    def __init__(self):

        super(ZeroLayer,self).__init__()

    def forward(self,x):

        return 0 * x
    
def get_vae(latent_dim,device):
    vae_decoder = nn.Sequential(nn.Linear(latent_dim,500),
                                nn.ReLU(),
                                nn.Linear(500,28**2),
                                nn.Sigmoid(),
                                nn.Unflatten(1,(1,28,28)))
    vae_shared_net = nn.Flatten(start_dim=1,end_dim=-1)
    vae_encoder_mu = nn.Sequential(nn.Linear(28**2,500),
                                nn.ReLU(),
                                nn.Linear(500,latent_dim))
    vae_encoder_d = nn.Sequential(nn.Linear(28**2,500),
                                nn.ReLU(),
                                nn.Linear(500,latent_dim))
    vae_encoder_L = ZeroLayer()
    vae_encoder = Encoder(vae_shared_net,
                        vae_encoder_mu,
                        vae_encoder_L,
                        vae_encoder_d,
                        latent_dim=latent_dim)
    vae = VAE(decoder=vae_decoder,encoder=vae_encoder,distribution=LowRankMultivariateNormal,device=device)
    
    return vae

def get_iwae(latent_dim,device):


    iwae_decoder = nn.Sequential(nn.Linear(latent_dim,500),
                                nn.ReLU(),
                                nn.Linear(500,28**2),
                                nn.Sigmoid(),
                                nn.Unflatten(1,(1,28,28)))
    
    iwae_shared_net = nn.Flatten(start_dim=1,end_dim=-1)
    iwae_encoder_mu = nn.Sequential(nn.Linear(28**2,500),
                                nn.ReLU(),
                                nn.Linear(500,latent_dim))
    iwae_encoder_d = nn.Sequential(nn.Linear(28**2,500),
                                nn.ReLU(),
                                nn.Linear(500,latent_dim))
    iwae_encoder_L = ZeroLayer()
        
    iwae_encoder = Encoder(iwae_shared_net,
                        iwae_encoder_mu,
                        iwae_encoder_L,
                        iwae_encoder_d,
                        latent_dim=latent_dim)
    
    iwae = IWAE(decoder=iwae_decoder,encoder=iwae_encoder,distribution=LowRankMultivariateNormal,device=device)

    return iwae