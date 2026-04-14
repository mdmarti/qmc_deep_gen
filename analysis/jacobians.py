from torch.func import jacfwd
from tqdm import tqdm
import torch
import numpy as np
from typing import Tuple
from scipy import ndimage

def get_norms_lattice(model:torch.nn.Module,lattice:torch.FloatTensor) -> Tuple[np.ndarray,np.ndarray]:
    """
    calculate the Frobenius norm of the jacobian for each point in latent space.
    This tells us the size of the change in data space for an infintessimal shift in
    latent space. It also gives us approximate category boundaries (areas where
    there are large visual changes in data space)

    model: QLVM
    lattice: QMC lattice over the latent space. I typically use a uniform grid 
    (as a uniform grid is a QMC lattice, just not an OPTIMAL lattice)

    returns
    norms: Frobenius norm of the Jacobian for the QLVM decoder
    log_norms: log Frobenius norms (+ small fudge factor to prevent log(0)). Use instead in cases
    where Frobenius norm has a very large range across the latent space, this will compress
    """
    
    norms = []
    forward_lam = lambda grid_point: model(grid_point.to(model.device),mod=False,random=False).flatten()

    def frob_norm_jac(grid_point):

        with torch.no_grad():
            jac = jacfwd(forward_lam)(grid_point).squeeze()
            #print(jac.shape)
            return torch.linalg.matrix_norm(jac).detach().cpu().numpy()**2

    for pt in tqdm(lattice,total=len(lattice)): #iterate point by point to keep computational demands small
        # this shouldn't take too long
        norm = frob_norm_jac(pt[None,:])
        norms.append(norm)

    norms = np.hstack(norms)
    log_norms = np.log(norms + 1e-12)

    return norms,log_norms



def sobel_x(img:np.ndarray)->np.ndarray:
    """
    sobel filter in the x direction of an image. this is an approximate edge detector, 
    along the x dimension.
    since our latent space is periodic, we wrap it.

    img: an mxn array. I use the norms over the latent lattice, or the log norms
    returns: sobel filtered image
    """

    return ndimage.convolve(img,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),mode='wrap')

def sobel_y(img:np.ndarray)->np.ndarray:
    """
    sobel filter in the y direction of an image. this is an approximate edge detector, 
    along the y dimension.
    since our latent space is periodic, we wrap it.

    img: an mxn array. I use the norms over the latent lattice, or the log norms
    returns: sobel filtered image
    """

    return ndimage.convolve(img,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T,mode='wrap')

def filter_img(img:np.ndarray,filter_type:str='sobel')->np.ndarray:
    """
    filters an image. two types are implemented. First, a sobel filter: this will find
    approximate edges. Combining an x and y sobel filter in this way allows us to find
    edges of all orientations. Second, a gaussian filter. this will instead blur the image.
    I primarily use this to reduce variability in the norms.

    img: an mxn array to be filtered
    filter_type: str, either `sobel` or `gaussian`.

    returns
    G, a filtered image
    """

    if filter_type == 'sobel':
        filt_y = sobel_y(img)
        filt_x = sobel_x(img)

        G = np.sqrt(filt_y**2 + filt_x**2)
    
    elif filter_type == 'gaussian':

        G = ndimage.gaussian_filter(img, 1,mode='wrap')
    

    else:
        raise NotImplementedError
    
    return G






