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
    def __init__(self, latent_dim=2, num_freqs=16, device=None):
        super(QMCLVM, self).__init__()

        self.latent_dim = latent_dim
        self.fourier_basis = FourierBasis(
            num_dims=latent_dim, num_freqs=num_freqs, device=device
        )

        # Decoder.
        #nn.Unflatten(1, (64, 7, 7)),
        #nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        #nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        #self.fourier_basis,
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim,2*num_freqs ** latent_dim),
            nn.Linear(2 * num_freqs ** latent_dim, 64*7*7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #nn.Linear(64*7*7,32*14*14),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),#nn.Linear(32*14*14,1*28*28),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, eval_grid):
        """
        eval_grid should be a sequence of `z`s that uniformly tile the latent space,
        and should be n_grid_points x latent_dim
        """
        r = torch.rand(1, self.latent_dim, device=eval_grid.device)
        x = (r + eval_grid) % 1
        return self.decoder(x)

    def latent_density(self,eval_grid,sample,lp_func,batch_size=-1,softmax=False):
        """
        eval_grid should be an n_grid_points x latent_dim sequence of latnet points,
        while sample should be n_data_points x n_channels x height x width
        """

        decoded = self.decoder(eval_grid %1)
        if batch_size == -1:
            batch_size = sample.shape[0]

        lps = []
        for on in range(0,sample.shape[0],batch_size):
            off = on + batch_size
            
            s = sample[on:off].tile(1,decoded.shape[0],1,1)
            d = decoded.swapaxes(0,1).tile(s.shape[0],1,1,1)
            
            lp = lp_func(d,s) #-1 * binary_cross_entropy(d,
                                    #            s,
                                    #            reduction='none'
                                    #           ).sum(axis=(2,3))
            lps.append(lp)
        lps = torch.cat(lps,axis=0)
        if softmax:
            return nn.Softmax(lps)
        
        return lps