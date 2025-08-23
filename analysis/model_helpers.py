import torch
from tqdm import tqdm
import numpy as np

def get_stacked_posterior(model,lattice,loader,lp):

    posteriors = []
    model.eval()
    for batch in tqdm(loader,total=len(loader)):
        data = batch[0].to(model.device)
        with torch.no_grad():
            posterior = model.posterior_probability(lattice.to(model.device),data,lp)
        posteriors.append(posterior.detach().cpu().numpy())
    
    stacked_posteriors = np.vstack(posteriors)

    return stacked_posteriors

def torus_forward(data):
    return np.concatenate([np.cos(2*np.pi*data),np.sin(2*np.pi*data)],axis=1)

def torus_reverse(data,dim=2):

    # tan = opp/adj = sin/cos
    # sin = opp/hyp
    # cos = adj/hyp

    # data[1,3,...,n] = sin(x)
    # data[0,2,...,n-1] = cos(x)
    
    angles = np.arctan2(data[:,dim:],data[:,:dim])
    angles[angles <0] = 2*np.pi + angles[angles < 0]

    return angles/(2*np.pi)