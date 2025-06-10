import torch
from torch.nn.functional import binary_cross_entropy
from torchvision.transforms import GaussianBlur

def binary_lp(samples, data):

    recon_loss = -1 * torch.mean(
        torch.special.logsumexp(
            torch.sum(
                -1 * binary_cross_entropy(
                    samples.swapaxes(0, 1).tile((data.shape[0], 1, 1, 1)),
                    data.tile(1, samples.shape[0], 1, 1),
                    reduction="none"
                ),
                axis=(2, 3)
            ),
            axis=1
        )
    )
    return recon_loss

def gaussian_lp(samples,data,var=1.):

    recon_loss = - torch.mean(
                    torch.special.logsumexp(
                        torch.sum(
                            -torch.nn.functional.gaussian_nll_loss(samples.swapaxes(0,1).tile((data.shape[0],1,1,1)),
                                                data.tile(1,samples.shape[0],1,1),
                                                var=var,
                                                reduction='none',
                                                ),
                            axis=(2,3)
                        ),
                        axis=1
                    )
                )
    return recon_loss

def gaussian_lp_with_blur(samples,data,var=1.,kernel_size=4,sigma=0.25):

    blur = GaussianBlur(kernel_size,sigma)

    blur_samples,blur_data = blur(samples),blur(data)

    return gaussian_lp(blur_samples,blur_data,var)


def gaussian_ELBO(reconstructions,distribution,targets,recon_precision=1e-2):


    B,h,w = reconstructions.shape
    d = h*w
    (mu,L,D) = distribution
    L = L.squeeze() # goes from B x d x 1 -> B x d

    err = targets - reconstructions
    neg_lp = torch.einsum('bhw,bhw->b',err,err) *(recon_precision)/2 + \
        d*torch.log(2*torch.pi)/2 - d * torch.log(recon_precision)/2 

    t12 = -1/2 *torch.log(D).sum(dim=-1) - 1/2*torch.log((1 + torch.einsum('bd,bd->b',L/D,L)))#torch.log(torch.prod(D,dim=-1)*(1 + torch.einsum('bd,bd->b',L/D,L)))
    t22 = 1/2 * (D.sum(dim=-1) + (L**2).sum(dim=-1))
    t32 = - d/2
    t42 = 1/2 * (mu**2).sum(dim=-1)

    kl = (t12 + t22 + t32 + t42)

    return neg_lp.mean(),kl.mean()


