import torch
from torch.nn.functional import binary_cross_entropy,gaussian_nll_loss
import numpy as np

############# For MNIST and Gerbil data ###########

def binary_lp(samples,data):

    ## following the example of torch BCEloss, this clamps the log terms at -100
    ## to prevent bad gradients
    ## Samples should be KSamples x Channels x H x W

    K,C,H,W = samples.shape
    
    samples = torch.clamp(samples,min=1e-6,max=1-1e-6)

    t1 = torch.einsum('bjdl,sjdl->bs',data,torch.log(samples))
    t2 = torch.einsum('bjdl,sjdl->bs',1-data,torch.log(1-samples))

    assert not torch.any(t1 == torch.nan)
    assert not (torch.any(t2 == torch.nan))

    ### returns: batch x n samples
    return (t1 + t2) 

def binary_evidence(samples, data,reduce=True):

    recon_loss = binary_lp(samples,data) #torch.cat(recon_loss,axis=1) 
    recon_loss = torch.special.logsumexp(
            recon_loss,
            axis=1
        )
    
    if reduce:
        return -1 * torch.mean(recon_loss)

    return -1* recon_loss


def binary_iwae_elbo(reconstructions,distribution,targets):

    z,dist = distribution
    B,k,d = z.shape
    rlp = torch.vmap(binary_cross_entropy,in_dims=(1,None),out_dims=1)
    recon_ll = -rlp(reconstructions,targets,reduction='none').sum(dim=(2,3,4))
    
    ### then, latent prior ll
    prior_ll =  -torch.einsum('kbd,kbd->bk',z,z)/2  - d*np.log(2*np.pi)/2 - d/2 ## needs to also be B x k
    
    ### finally, learned latent dist ll
    latent_ll = dist.log_prob(z).permute(1,0) # should be B x k
    

    return -torch.special.logsumexp(recon_ll + prior_ll - latent_ll,dim=1).mean(dim=0) + np.log(k),torch.tensor([0.]).to(reconstructions.device)


############ for all other datasets #########

def gaussian_lp(samples,data,var):

    """
    expects samples to be 
    S x 1 x D x D
    expects data to be
    B x 1 x D x D
    """
    K,C,H,W = samples.shape
    if C == H:# if channels are in the wrong position
        samples = samples.permute(0,3,1,2)
        data = data.permute(0,3,1,2)
    
    vmapped_lp = torch.vmap(torch.vmap(gaussian_nll_loss,in_dims=(0,None)),in_dims=(None,0))
    return -vmapped_lp(samples,data,var=var,reduction='sum',full=True) # since this should be log(p(x|z)p(z))


def gaussian_evidence(samples,data,var,reduce=True):


    recon_loss = gaussian_lp(samples,data,var)

    recon_loss = torch.special.logsumexp(
            recon_loss,
            axis=1
        )
    if reduce:
        return -1 * torch.mean(recon_loss)

    return -1* recon_loss


def gaussian_elbo(reconstructions,distribution,targets,recon_precision=1e-2,beta=1):


    B,c,h,w = reconstructions.shape
    d = c*h*w
    (mu,L,D) = distribution
    L = L.squeeze(-1) # goes from B x d x 1 -> B x d

    z_dim = mu.shape[1]
    
    neg_lp = gaussian_nll_loss(reconstructions,
                                targets,
                                var=1/recon_precision,
                                reduction='none',
                                full=True
                                ).sum(axis=(1,2,3))


    t12 = -1/2 *torch.log(D).sum(dim=-1) - 1/2*torch.log((1 + torch.einsum('bd,bd->b',L/D,L)))
    t22 = 1/2 * (D.sum(dim=-1) + (L**2).sum(dim=-1))
    t32 = - z_dim/2
    t42 = 1/2 * (mu**2).sum(dim=-1)

    kl = beta*(t12 + t22 + t32 + t42)

    return neg_lp.mean(),kl.mean()

def gaussian_iwae_elbo(reconstructions,distribution,targets,recon_precision=1e-2):

    """
    here, reconstructions should be: B x k x C x H x W 
    where k is the number of reconstructions, B is batch size,
    C,H,W are data dimensions.

    distribution should be a tuple, with the first element corresponding to samples from dist
    of shape B x k x d (d = latent dimensionality)

    the second element should be the latent distribution object, which has a .log_prob method

    targets are data, and should be BxCxHxW
    """

    if len(targets.shape) == len(reconstructions.shape):
        targets = targets.squeeze(1)
    z,dist = distribution
    B,k,d = z.shape
    
    ### first, reconstruction ll. 
    rlp = torch.vmap(gaussian_nll_loss,in_dims=(1,None),out_dims=1)
    recon_ll = -rlp(reconstructions,targets,var=1/recon_precision,reduction='none',full=True).sum(dim=(2,3,4))
    ### then, latent prior ll
    prior_ll =  -torch.einsum('kbd,kbd->bk',z,z)/2  - d*np.log(2*np.pi)/2 - d/2 ## needs to also be B x k
    
    ### finally, learned latent dist ll
    latent_ll = dist.log_prob(z).permute(1,0) # should be B x k
    
    return -torch.special.logsumexp(recon_ll + prior_ll - latent_ll,dim=1).mean(dim=0) + np.log(k),torch.tensor([0.]).to(reconstructions.device)
