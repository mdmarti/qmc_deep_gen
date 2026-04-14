import torch
from tqdm import tqdm
import numpy as np
from collections.abc import Callable

def get_stacked_posterior(model:torch.nn.Module,
                          lattice:torch.FloatTensor,
                          loader:torch.utils.data.DataLoader,
                          lp:Callable)->np.ndarray:
    """
    gets the posterior distribution over a lattice for an entire dataloader.
    returns this posterior as an ndarray. the current implementation
    is for unconditional models, to extend to conditional
    we would need to update qlvm.posterior_probability to include conditional info

    model: QLVM
    lattice: QMC lattice
    loader: Dataloader, containing the data you want to embed.
    lp: reconstruction log probability,
        used to calculate posterior probabilities

    returns: stacked posterior, the posteriors over the lattice for each data sample 
    """

    posteriors = []
    lattice = lattice.to(model.device)
    (M,D) = lattice.shape #since lattice should be n samples x dimension
    N = len(loader.dataset) # total number of samples
    model.eval()
    for batch in tqdm(loader,total=len(loader)):
        data = batch[0].to(model.device)
        total_samples += len(data)
        with torch.no_grad():
            posterior = model.posterior_probability(lattice,data,lp)
            assert posterior.shape[0] == len(data),print(posterior.shape)
            assert posterior.shape[1] == M,print(posterior.shape)
            assert len(posterior.shape) == 2, print(posterior.shape)
        posteriors.append(posterior.detach().cpu().numpy())
    
    stacked_posteriors = np.vstack(posteriors) 
    assert stacked_posteriors.shape[0] == N, print(stacked_posteriors.shape)
    assert stacked_posteriors.shape[1] == M, print(stacked_posteriors.shape)
    assert len(stacked_posteriors.shape) == 2, print(stacked_posteriors.shape)

    return stacked_posteriors

def torus_forward(data:np.ndarray) -> np.ndarray:
    """
    passes data forwards through a torus (sin,cosine) basis

    data: elements in latent space

    returns:
        data embedded through sin, cos basis
    """
    return np.concatenate([np.cos(2*np.pi*data),np.sin(2*np.pi*data)],axis=1)

def torus_reverse(data:np.ndarray,dim:int=2)->np.ndarray:

    """
    passes data backwards through a torus (sin, cosine) basis
    since our latent space is constrained to (0,1), we can do this

    data: elements embedded in sine, cosine basis
    dim: original latent dimension
    returns:
        data re-embedded in latent space -- in 0-1 coordinates
    """
    # we love trigonometry! a quick reminder
    # tan = opp/adj = sin/cos
    # sin = opp/hyp
    # cos = adj/hyp

    # data[0,1,...dim-1] = cos(x)
    # data[dim,dim+1,...,2*dim -1] = sin(x)
    
    # we use arctan2 to retrieve the correct quadrant
    angles = np.arctan2(data[:,dim:],data[:,:dim])
    angles[angles <0] = 2*np.pi + angles[angles < 0]

    return angles/(2*np.pi)
