import torch
from torch.nn.functional import binary_cross_entropy

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

def gaussian_lp(samples,data,var=1):

    recon_loss = - torch.mean(
                    torch.special.logsumexp(
                        torch.sum(
                            -torch.nn.gaussian_nll_loss(samples.swapaxes(0,1).tile((data.shape[0],1,1,1)),
                                                data.tile(1,samples.shape[0],1,1),
                                                var=var,
                                                reduction=None
                                                ),
                            axis=(2,3)
                        ),
                        axis=1
                    )
                )
    return recon_loss

