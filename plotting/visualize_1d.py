import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plotting.visualize import format_plot_axis

def model_grid_plot(model,n_samples_dim,fn='',show=True,origin=None,cm='grey',model_type='qmc'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_samples_dim = 10
    n_samples=n_samples_dim
    cmap=mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(-1,n_samples)
    with torch.no_grad():
        #z = torch.rand(n_samples, 2).to(device)
        #xx,yy = torch.meshgrid([torch.linspace(EPS1,1-EPS2,n_samples_dim)]*2,indexing='ij')
        z = torch.linspace(0,1,n_samples).to(device)[:,None]
        
        sample = model.decoder(z).detach().cpu()
    z = z.detach().cpu().numpy()
    inds = np.arange(n_samples)
    cs = cmap(norm(inds))

    
    mosaic = [[f"sample {ii}" for ii in range(n_samples_dim)]]               
    

    fig, axes = plt.subplot_mosaic(mosaic,figsize=(30,10),sharex=True,sharey=True,gridspec_kw={'wspace':0.01,'hspace':0.01})

    for ii in range(n_samples):
        ax = axes[f"sample {ii}"]
        ax.imshow(sample[ii, 0, :, :], cmap=cm,origin=origin)
        ax.spines[['right','left','top','bottom']].set_color(cmap(norm(ii)))
        ax.spines[['right','left','top','bottom']].set_linewidth(4)
        ax.set_yticks([])
        ax.set_xticks([])


    if show:
        plt.show()
    else:
        plt.savefig(fn)
    plt.close()

def density_plot(densities,labels,fn='density_1d.svg',show=False):

    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5),)

    ls = []
    for density,label in zip(densities,labels,width_ratios=(5,1)):

        xax = np.linspace(0,1,len(density))
        l,=axs[0].plot(xax,density,label=label)
        ls.append(l)

    axs[1].legend(ls,labels,frameon=False)
    axs[1].spines[['left','right','top','bottom']].set_visible(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[0] = format_plot_axis(axs[0],xlabel="Latent axis",ylabel="Density",xticks=[0,0.5,1.0])

    if show:
        plt.show()

    else:
        plt.savefig(fn)

    plt.close()
