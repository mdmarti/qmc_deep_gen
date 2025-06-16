import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

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
    

class QMCLVM(nn.Module):
    def __init__(self, latent_dim=2,device=None,decoder=None):
        super(QMCLVM, self).__init__()

        self.latent_dim = latent_dim
        #self.fourier_basis = FourierBasis(
        #    num_dims=latent_dim, num_freqs=num_freqs, device=device
        #)

        # Decoder.
        #nn.Unflatten(1, (64, 7, 7)),
        #nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        #nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        #self.fourier_basis,
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim,2048),
            nn.Linear(2048, 64*7*7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #nn.Linear(64*7*7,32*14*14),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),#nn.Linear(32*14*14,1*28*28),
            nn.Sigmoid(),
        ).to(device) if decoder is None else decoder.to(device)

    def forward(self, eval_grid,random=True):
        """
        eval_grid should be a sequence of `z`s that uniformly tile the latent space,
        and should be n_grid_points x latent_dim
        """
        r = torch.rand(1, self.latent_dim, device=eval_grid.device) if random else torch.zeros((1,self.latent_dim),device=eval_grid.device)
        x = (r + eval_grid) % 1
        return self.decoder(x)


    def posterior_probability(self,grid,data,log_likelihood,batch_size=32):

        """
        log likelihood should include the summation over data dimensions
        """
        preds = self.decoder(grid)

        model_grid_lls = []
        N = data.shape[0]
        for on_ind in range(0,N,batch_size):

            off_ind = min(N,on_ind + batch_size)
            sample = data[on_ind:off_ind]
            model_grid_lls.append(log_likelihood(preds,sample))
            
        model_grid_lls = torch.cat(model_grid_lls,dim=0) #each entry A_ij is log p(x_i|z_j)
        ## as such, model_Grid_array should be n_data x n_grid points
        ll_per_grid = model_grid_lls.sum(dim=0)
        evidence = torch.special.logsumexp(model_grid_lls,dim=1).sum(dim=0)

        posterior = ll_per_grid - evidence

        return nn.Softmax(dim=0)(posterior)
    
class Torus_QMCLVM(QMCLVM):

    def __init__(self, latent_dim=2,device=None,decoder=None):

        self.latent_dim = latent_dim
        super(Torus_QMCLVM,self).__init__(latent_dim=latent_dim,device=device,decoder=decoder)

    def forward(self,eval_grid):

        r = torch.rand(1, self.latent_dim, device=eval_grid.device)
        x = (r + eval_grid) % 1
        theta_x = 2*torch.pi * x 
        torus_basis = torch.cat([torch.cos(theta_x),torch.sin(theta_x)],dim=-1)
        return self.decoder(torus_basis)/2 + self.decoder(2 * torch.pi - torus_basis)/2 


    
