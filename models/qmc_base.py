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
    

class TorusBasis(nn.Module):

    def __init__(self):

        super(TorusBasis,self)._init__()
        
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

    def forward(self, eval_grid,random=True):
        """
        eval_grid should be a sequence of `z`s that uniformly tile the latent space,
        and should be n_grid_points x latent_dim
        """
        r = torch.rand(1, self.latent_dim, device=self.device) if random else torch.zeros((1,self.latent_dim),device=self.device)
        x = (r + eval_grid) % 1
        return self.decoder(x)


    def posterior_probability(self,grid,data,log_likelihood,batch_size=32):

        """
        log likelihood should include the summation over data dimensions
        """
        grid = grid.to(self.device)
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
    



    
