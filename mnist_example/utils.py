from torch import nn
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from vae import *
from qlvm import *


def get_models(vae_latent_dim,iwae_latent_dim,device):


    qlvm_latent_dim=2
    ##vae_latent_dim=2
    ##iwae_latent_dim=2

    qlvm_decoder = nn.Sequential(nn.Linear(2*qlvm_latent_dim,500),
                                nn.ReLU(),
                                nn.Linear(500,28**2),
                                nn.Sigmoid(),
                                nn.Unflatten(1,(1,28,28)))

    vae_decoder = nn.Sequential(nn.Linear(vae_latent_dim,500),
                                nn.ReLU(),
                                nn.Linear(500,28**2),
                                nn.Sigmoid(),
                                nn.Unflatten(1,(1,28,28)))
    iwae_decoder = nn.Sequential(nn.Linear(iwae_latent_dim,500),
                                nn.ReLU(),
                                nn.Linear(500,28**2),
                                nn.Sigmoid(),
                                nn.Unflatten(1,(1,28,28)))

    vae_shared_net = nn.Flatten(start_dim=1,end_dim=-1)
    vae_encoder_mu = nn.Sequential(nn.Linear(28**2,500),
                                nn.ReLU(),
                                nn.Linear(500,vae_latent_dim))
    vae_encoder_d = nn.Sequential(nn.Linear(28**2,500),
                                nn.ReLU(),
                                nn.Linear(500,vae_latent_dim))
    vae_encoder_L = ZeroLayer()
    vae_encoder = Encoder(vae_shared_net,
                        vae_encoder_mu,
                        vae_encoder_L,
                        vae_encoder_d,
                        latent_dim=vae_latent_dim)

    iwae_shared_net = nn.Flatten(start_dim=1,end_dim=-1)
    iwae_encoder_mu = nn.Sequential(nn.Linear(28**2,500),
                                nn.ReLU(),
                                nn.Linear(500,iwae_latent_dim))
    iwae_encoder_d = nn.Sequential(nn.Linear(28**2,500),
                                nn.ReLU(),
                                nn.Linear(500,iwae_latent_dim))
    iwae_encoder_L = ZeroLayer()
        
    iwae_encoder = Encoder(iwae_shared_net,
                        iwae_encoder_mu,
                        iwae_encoder_L,
                        iwae_encoder_d,
                        latent_dim=iwae_latent_dim)
    
    qlvm = QLVM(decoder=qlvm_decoder,device=device,latent_dim=2,basis=TorusBasis())
    vae = VAE(decoder=vae_decoder,encoder=vae_encoder,distribution=LowRankMultivariateNormal,device=device)
    iwae = IWAE(decoder=iwae_decoder,encoder=iwae_encoder,distribution=LowRankMultivariateNormal,device=device)

    return qlvm,vae,iwae