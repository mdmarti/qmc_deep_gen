import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import string 

class FourierBasis(nn.Module):

    def __init__(self, num_dims=2, num_freqs=4, device=None):
        super(FourierBasis, self).__init__()

        # F.shape == (num_dims x num_basis_functions)
        self.F = 2 * torch.pi * (
            torch.stack(
                torch.meshgrid(
                    [torch.arange(num_freqs)] * num_dims, indexing="ij"
                )
            ).reshape(
                num_dims, num_freqs ** num_dims
            )
        ).to(device)
        # self.wsin = nn.Parameter(torch.ones(num_freqs ** num_dims))
        # self.wcos = nn.Parameter(torch.ones(num_freqs ** num_dims))

    def forward(self, x):
        """
        x.shape == (batch_size, num_dims)
        """
        return torch.hstack(
            (torch.sin(x @ self.F), torch.cos(x @ self.F))
        )
    

class TorusBasis(nn.Module):

    def __init__(self):

        super(TorusBasis,self).__init__()
        
    def forward(self,data):
        
        return torch.cat([torch.cos(2*torch.pi*data),torch.sin(2*torch.pi*data)],dim=1)

class QMCLVM(nn.Module):
    def __init__(self, latent_dim=2,device=None,decoder=None):
        super(QMCLVM, self).__init__()
        """
        if you want a fourier basis, you'd better put it in the gosh dang de coder!
        """
        self.device=device

        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim,2048),
            nn.Linear(2048, 64*7*7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #nn.Linear(64*7*7,32*14*14),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),#nn.Linear(32*14*14,1*28*28),
            nn.Sigmoid(),
        ).to(device) if decoder is None else decoder.to(device)

    def forward(self, eval_grid,random=True,mod=True):
        """
        eval_grid should be a sequence of `z`s that uniformly tile the latent space,
        and should be n_grid_points x latent_dim
        """
        r = torch.rand(1, self.latent_dim, device=self.device) if random else torch.zeros((1,self.latent_dim),device=self.device)
        x = (r + eval_grid) % 1 if mod else r+eval_grid
        return self.decoder(x)


    def posterior_probability(self,grid,data,log_likelihood):

        """
        this needs to give posterior for each point individually
        """
        """
        log likelihood should include the summation over data dimensions
        """
    
        with torch.no_grad():
            preds = self.decoder(grid %1)
    
            model_grid_lls = log_likelihood(preds,data) #each entry A_ij is log p(x_i|z_j)
                
            ## as such, model_Grid_array should be n_data x n_grid points
            #ll_per_grid = model_grid_lls.sum(dim=0)
            evidence = torch.special.logsumexp(model_grid_lls,dim=1,keepdims=True) ## n_data x 1
            
            posterior = model_grid_lls - evidence

            return nn.Softmax(dim=1)(posterior) # posterior over grid points for each sample
    
    def round_trip(self,grid,data,log_likelihood,recon_type='posterior',n_samples=25):

        grid = grid.to(self.device)

        if recon_type == 'rqmc':
            posterior_grid = []
            for _ in range(n_samples):
                tmp_grid = (grid + torch.rand((1,2),device=self.device)) % 1
                posterior = self.posterior_probability(tmp_grid,data,log_likelihood)
                posterior_grid.append(posterior.to(self.device) @ tmp_grid)
            posterior_grid = torch.stack(posterior_grid,axis=0).mean(axis=0)
        else:
            posterior = self.posterior_probability(grid,data,log_likelihood)
            posterior = posterior.to(self.device)
        
        
        if (recon_type == 'posterior') or (recon_type=='rqmc'):
            posterior_grid = posterior @ (grid % 1)
        elif recon_type == 'argmax':
            posterior_grid = grid[torch.argmax(posterior)][None,:] % 1
        elif recon_type == 'rqmc':
            pass
        else:
            raise NotImplementedError
        recon = self.decoder(posterior_grid)

        return recon
    
    def embed_data(self,grid,loader,log_likelihood,embed_type='posterior',n_samples=10):

        latents = []
        labels = [] 
        grid = grid.to(self.device)
        with torch.no_grad():
            for (data,label) in tqdm(loader,desc='embedding latents',total=len(loader)):
                data = data.to(self.device).to(torch.float32)
                if type(label) == tuple:
                    labels.append([string.ascii_lowercase.index(l.lower()[0]) for l in label])
                else:
                    labels.append(label.detach().cpu().numpy())

                if embed_type == 'rqmc':
                    latent_batch = []

                    for _ in range(n_samples):
                        tmp_grid = (grid + torch.rand((1,2),device=self.device))%1
                        posterior = self.posterior_probability(tmp_grid,data,log_likelihood) # Bsz x Grid size
                        latent_batch.append(posterior.to(self.device) @ tmp_grid) # Bsz x latent dim
                    latent_batch = torch.stack(latent_batch,axis=0).mean(axis=0) # Bsz x latent dim
                    latents.append(latent_batch.detach().cpu())
                elif embed_type == 'posterior':
                    posterior = self.posterior_probability(grid,data,log_likelihood)
                    # posterior is B x S, convert to B x 2 for weighted grid
                    latents.append((posterior @ (grid%1)).detach().cpu())
                elif embed_type == 'argmax':
                    posterior = self.posterior_probability(grid,data,log_likelihood)
                    max_inds = torch.argmax(posterior,axis=1)
                    latents.append((grid[max_inds]%1).detach().cpu()) # this may work? double check

        latents = torch.vstack(latents).detach().cpu().numpy()
        labels = np.hstack(labels)
        return latents,labels
    



    
