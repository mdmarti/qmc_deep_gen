import torch
from torch.nn.functional import binary_cross_entropy
from torchvision.transforms import GaussianBlur
import numpy as np

def binary_evidence(samples, data):
    # (should be sum for full, joint evidence)
    # but we can also do expected evidence per sample
    recon_loss = -1 * torch.mean(
        torch.special.logsumexp(
            torch.sum(binary_lp(samples,data),
                axis=(2, 3)
            ),
            axis=1
        )
    )
    return recon_loss

def gaussian_evidence(samples,data,var=1.):

    recon_loss = - torch.mean(
                    torch.special.logsumexp(
                        torch.sum(gaussian_lp(samples,data,var),
                            axis=(2,3)
                        ),
                        axis=1
                    )
                )
    return recon_loss

def gaussian_evidence_with_blur(samples,data,var=1.,kernel_size=4,sigma=0.25):

    blur = GaussianBlur(kernel_size,sigma)

    blur_samples,blur_data = blur(samples),blur(data)

    return gaussian_evidence(blur_samples,blur_data,var)

def binary_lp(samples,data):

    """
    expects data to be:
    B x 1 x H x W
    expects samples to be:
    1 X S x H x W
    """
    -1 * binary_cross_entropy(
                    samples.swapaxes(0, 1).tile((data.shape[0], 1, 1, 1)),
                    data.tile(1, samples.shape[0], 1, 1),
                    reduction="none"
                )

def gaussian_lp(samples,data,var=1):
    """
    expects data to be:
    B x 1 x H x W
    expects samples to be:
    1 X S x H x W
    """

    return -torch.nn.functional.gaussian_nll_loss(samples.swapaxes(0,1).tile((data.shape[0],1,1,1)),
                                                data.tile(1,samples.shape[0],1,1),
                                                var=var,
                                                reduction='none',
                                                full=True
                                                )


def gaussian_ELBO(reconstructions,distribution,targets,recon_precision=1e-2):


    """
    to do: error proof this
    """

    B,c,h,w = reconstructions.shape
    d = c*h*w
    (mu,L,D) = distribution
    L = L.squeeze(-1) # goes from B x d x 1 -> B x d

    z_dim = mu.shape[1]
    #err = targets - reconstructions
    neg_lp = torch.nn.functional.gaussian_nll_loss(targets,
                                                   reconstructions,
                                                   var=1/recon_precision,
                                                   reduction='none',
                                                   full=True
                                                   ).sum(axis=(1,2,3))
    #torch.einsum('bchw,bchw->b',err,err) *(recon_precision)/2 + \
        #d*np.log(2*torch.pi)/2 - d * np.log(recon_precision)/2 

    t12 = -1/2 *torch.log(D).sum(dim=-1) - 1/2*torch.log((1 + torch.einsum('bd,bd->b',L/D,L)))#torch.log(torch.prod(D,dim=-1)*(1 + torch.einsum('bd,bd->b',L/D,L)))
    t22 = 1/2 * (D.sum(dim=-1) + (L**2).sum(dim=-1))
    t32 = - z_dim/2
    t42 = 1/2 * (mu**2).sum(dim=-1)

    kl = (t12 + t22 + t32 + t42)

    return neg_lp.mean()+kl.mean()



