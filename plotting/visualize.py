import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import torch


def model_grid_plot(model,n_samples_dim,fn='',show=True,origin=None,cm='grey'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_samples_dim = 10
    n_samples=n_samples_dim**2
    cmap=mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(-1,n_samples)
    with torch.no_grad():
        #z = torch.rand(n_samples, 2).to(device)
        xx,yy = torch.meshgrid([torch.arange(int(np.sqrt(n_samples)))/int(np.sqrt(n_samples))]*2,indexing='ij')
        z = torch.stack([xx.flatten(),yy.flatten()],axis=-1).to(device)
        sample = model.decoder(z).detach().cpu()
    z = z.detach().cpu().numpy()
    inds = np.arange(n_samples)
    cs = cmap(norm(inds))

    
    mosaic = [['scatter grid']*n_samples_dim]*n_samples_dim +\
              [[f"sample {ii*n_samples_dim + jj}" for ii in range(n_samples_dim)] for jj in range(n_samples_dim)]
               
    

    fig, axes = plt.subplot_mosaic(mosaic,figsize=(20,20))

    for ii in range(n_samples):
        ax = axes[f"sample {ii}"]
        ax.imshow(sample[ii, 0, :, :], cmap=cm,origin=origin)
        ax.spines[['right','left','top','bottom']].set_color(cmap(norm(ii)))
        ax.spines[['right','left','top','bottom']].set_linewidth(4)
        ax.set_yticks([])
        ax.set_xticks([])

    axes['scatter grid'].scatter(z[:,0],z[:,1],c=cs)
    axes['scatter grid'].set_xlim([-0.05,1-0.01*n_samples_dim])
    axes['scatter grid'].set_ylim([-0.05,1-0.01*n_samples_dim])
    axes['scatter grid'].set_xticks([])
    axes['scatter grid'].set_yticks([])
    axes['scatter grid'].spines[['right','left','top','bottom']].set_visible(False)
    if show:
        plt.show()
    else:
        plt.savefig(fn)
    plt.close()




def class_density_plot(grid,density,figname,fn='',show=True,ax=None):

    if ax is None:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10))

    ax.scatter(grid[:,0],grid[:,1],c=density)
    ax.set_title(figname)

    if ax is None:
        if show:
            plt.show()
        else:
            plt.savefig(fn)
        plt.close()
        return
    return ax






    

