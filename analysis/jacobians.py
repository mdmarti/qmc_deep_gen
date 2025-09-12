from torch.autograd.functional import jacobian
from torch.func import jacfwd,vmap,jacrev
from tqdm import tqdm
import torch
import numpy as np

from scipy import ndimage

def get_norms_lattice(model,lattice):

    
    norms = []
    forward_lam = lambda grid_point: model(grid_point.to(model.device),mod=False,random=False).flatten()

    def frob_norm_jac(grid_point):

        with torch.no_grad():
            jac = jacfwd(forward_lam)(grid_point).squeeze()
            #print(jac.shape)
            return torch.linalg.matrix_norm(jac).detach().cpu().numpy()**2

    for pt in tqdm(lattice,total=len(lattice)):
        norm = frob_norm_jac(pt[None,:])
        norms.append(norm)

    norms = np.hstack(norms)
    log_norms = np.log(norms + 1e-12)

    return norms,log_norms



def sobel_x(img):

    return ndimage.convolve(img,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),mode='wrap')

def sobel_y(img):

    return ndimage.convolve(img,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T,mode='wrap')

def filter_img(img,filter_type='sobel'):

    if filter_type == 'sobel':
        filt_y = sobel_y(img)
        filt_x = sobel_x(img)

        G = np.sqrt(filt_y**2 + filt_x**2)
    
    elif filter_type == 'gaussian':

        G = ndimage.gaussian_filter(img, 1,mode='wrap')

    return G






