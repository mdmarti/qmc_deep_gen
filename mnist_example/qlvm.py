import torch
from torch import nn
import numpy as np
from tqdm import tqdm


##### Basis functions over the latent space #########

class TorusBasis(nn.Module):

    def __init__(self):
        """
        enforces periodicity over the latent space
        """

        super(TorusBasis,self).__init__()
        
    def forward(self,data):
        
        return torch.cat([torch.cos(2*torch.pi*data),torch.sin(2*torch.pi*data)],dim=-1)
    
    def reverse(self,data):

        d = int(data.shape[-1]//2)
        angles = torch.atan2(data[:,d:],data[:,:d])
        angles[angles < 0] = torch.pi*2 + angles[angles < 0]

        return angles/(2*torch.pi)

class IdentityBasis(nn.Module):

    def __init__(self):

        super (IdentityBasis,self).__init__()

    def forward(self,data):

        return data
    
    def reverse(self,data):

        return data
    
class GaussianICDFBasis(nn.Module):

    def __init__(self,device='cuda'):

        super(GaussianICDFBasis,self).__init__()
        self.device=device
        self.dist = torch.distributions.Normal(torch.tensor([[0.,]],device=device),torch.tensor([[1.,]],device=device))
        self.icdf = lambda x: self.dist.icdf(torch.clip(x,min=1e-4,max=1-1e-4))
        self.cdf = self.dist.cdf
        
    def forward(self,data):
        
        return self.icdf(data)
    
    def reverse(self,data):

        return self.cdf(data)
    
class QLVM(nn.Module):
    def __init__(self, decoder,device,latent_dim=2,basis=TorusBasis()):
        super(QLVM, self).__init__()
        """
        requires:
            decoder (nn.Module): maps from latent space to data space
            device (nn.device() or str): device to train on
            latent_dim (int): dimension of the latent space
            basis (nn.Module): map to pass latent lattice through 
                            feeding to decoder
        """


        self.device=device

        self.latent_dim = latent_dim
        self.basis = basis
        self.decoder = decoder

        self.to(device) 


    def forward(self, lattice,random=True,mod=True):
        """
        eval_grid should be a sequence of `z`s that uniformly tile the latent space,
        and should be n_grid_points x latent_dim
        """
        

        r = torch.rand(1, self.latent_dim, device=self.device) if random else torch.zeros((1,self.latent_dim),device=self.device)
        x = (r + lattice) % 1 if mod else r+lattice
        basis = self.basis(x)
        
        return self.decoder(basis)


    def posterior_probability(self,lattice,data,log_likelihood):
        """
        takes as input: 
            lattice (torch.Tensor): QMC lattice over the latent space
            data (torch.Tensor): data to find posterior over lattice for
            log_likelihood (function): log likelihood function used to train the model
        """
        
    
        with torch.no_grad():
            basis = self.basis(lattice % 1)
            
            preds = self.decoder(basis)
    
            model_lattice_lls = log_likelihood(preds,data) #each entry A_ij is log p(x_i|z_j)
                
            ## as such, model_lattice_lls should be n_data x n_grid points
            evidence = torch.special.logsumexp(model_lattice_lls,dim=1,keepdims=True) - np.log(len(basis)) ## n_data x 1
            
            posterior = model_lattice_lls - evidence

            return nn.Softmax(dim=1)(posterior) # posterior over grid points for each sample
    
    def round_trip(self,grid,data,log_likelihood,recon_type='posterior',n_samples=10):

        grid = grid.to(self.device)
        with torch.no_grad():
            if recon_type == 'rqmc':
                posterior_grid = []

                for _ in range(n_samples):
                    tmp_grid = (grid + torch.rand((1,grid.shape[1]),device=self.device))%1
                    posterior = self.posterior_probability(tmp_grid,data,log_likelihood) # Bsz x Grid size
                    posterior_grid.append(self.basis.reverse(
                                            posterior.to(self.device) @ self.basis.forward(tmp_grid)
                     )) # Bsz x latent dim
                posterior_grid = self.basis.reverse(self.basis.forward(torch.stack(posterior_grid,axis=0)).mean(axis=0))
            elif recon_type == 'rqmc_recon':
                posterior_ims = []

                for _ in range(n_samples):
                    tmp_grid = (grid + torch.rand((1,grid.shape[1]),device=self.device)) % 1
                    posterior = self.posterior_probability(tmp_grid,data,log_likelihood)
                    recons = self.decoder(tmp_grid) # G x C x H x W (or B)
                    recons = torch.einsum('BG,GCHW->BCHW',posterior,recons)#posterior.to(self.device) @ recons
                    posterior_ims.append(recons)
                recon = torch.stack(posterior_ims,axis=0).mean(axis=0)

            else:
                posterior = self.posterior_probability(grid,data,log_likelihood)
                posterior = posterior.to(self.device)
            
            if 'argmax' in recon_type:
                """
                same if we do in image space vs. latent space
                """
                
                posterior_grid = grid[torch.argmax(posterior)][None,:] % 1
                
                recon = self.forward(posterior_grid,random=False,mod=False)

            elif ('recon' not in recon_type):
                """
                different in latent space
                """
                if (recon_type == 'posterior'):
                    posterior_grid = self.basis.reverse(
                                    posterior.to(self.device) @ self.basis.forward(grid % 1)
                     )
                
                elif recon_type == 'rqmc':
                    pass
                else:
                    raise NotImplementedError
                recon = self.forward(posterior_grid,random=False,mod=False)
            else:
                if 'posterior' in recon_type:
                    recons = self.forward(grid,random=False,mod=True)
                    recon =  torch.einsum('BG,GCHW->BCHW',posterior,recons)
                elif 'rqmc' in recon_type:
                    pass 
                else:
                    raise NotImplementedError

        return recon
    
    def embed_data(self,grid,loader,log_likelihood,embed_type='posterior',n_samples=10):
        """
        embeds all data in a dataloader
        """

        latents = []
        labels = [] 
        grid = grid.to(self.device)
        with torch.no_grad():
            for (data,label) in tqdm(loader,desc='embedding latents',total=len(loader)):
                data = data.to(self.device).to(torch.float32)

                labels.append(label.detach().cpu().numpy())

                if embed_type == 'rqmc':
                    latent_batch = []

                    for _ in range(n_samples):
                        tmp_grid = (grid + torch.rand((1,2),device=self.device))%1
                        posterior = self.posterior_probability(tmp_grid,data,log_likelihood) # Bsz x Grid size
                        latent_batch.append(self.basis.reverse(
                                            posterior.to(self.device) @ self.basis.forward(tmp_grid)
                        )) # Bsz x latent dim
                    latent_batch = self.basis.reverse(self.basis.forward(torch.stack(latent_batch,axis=0)).mean(axis=0)) # Bsz x latent dim
                    latents.append(latent_batch.detach().cpu())
                elif embed_type == 'posterior':
                    posterior = self.posterior_probability(grid,data,log_likelihood)
                    # posterior is B x S, convert to B x 2 for weighted grid
                    latents.append(self.basis.reverse(
                                            posterior @ self.basis.forward(grid)
                     ).detach().cpu())
                elif embed_type == 'argmax':
                    posterior = self.posterior_probability(grid,data,log_likelihood)
                    max_inds = torch.argmax(posterior,axis=1)
                    latents.append((grid[max_inds]%1).detach().cpu()) # this may work? double check

        latents = torch.vstack(latents).detach().cpu().numpy()
        labels = np.hstack(labels)
        return latents,labels
    
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a

def gen_fib_basis(m):
    """
    Creates random numbers tiling a cube [0,1]^2 where m is element of the fibonacci sequence
    """

    n = fib(m)
    z = torch.tensor([1.,fib(m-1)])

    return torch.arange(0,n)[:,None]*z[None,:]/n

def get_qlvm(device):


    qlvm_decoder = nn.Sequential(nn.Linear(4,500),
                                nn.ReLU(),
                                nn.Linear(500,28**2),
                                nn.Sigmoid(),
                                nn.Unflatten(1,(1,28,28)))
    qlvm = QLVM(decoder=qlvm_decoder,device=device,latent_dim=2,basis=TorusBasis())

    return qlvm
