import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

EPS1 = 1e-15
EPS2 = 1e-6
def model_grid_plot(model,n_samples_dim,fn='',show=True,origin=None,cm='grey',model_type='qmc'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_samples_dim = 10
    n_samples=n_samples_dim**2
    cmap=mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(-1,n_samples)
    with torch.no_grad():
        #z = torch.rand(n_samples, 2).to(device)
        xx,yy,zz = torch.meshgrid([torch.linspace(0,1,n_samples_dim)]*3,indexing='ij')
        for z_index in range(n_samples_dim):
            z = torch.stack([xx[:,:,z_index].flatten(),yy[:,:,z_index].flatten(),zz[:,:,z_index].flatten()],axis=-1).to(device)
        
        #z = torch.linspace(0,1,n_samples).to(device)[:,None]
        
            sample = model.decoder(z).detach().cpu()
            mosaic = [[f"sample {ii*n_samples_dim + jj}" for ii in range(n_samples_dim)] for jj in range(n_samples_dim)]                

    

            fig, axes = plt.subplot_mosaic(mosaic,figsize=(20,20),sharex=True,sharey=True,gridspec_kw={'wspace':0.01,'hspace':0.01})
        
            for ii in range(n_samples):
                ax = axes[f"sample {ii}"]
                ax.imshow(sample[ii, 0, :, :], cmap=cm,origin=origin)
                ax.spines[['right','left','top','bottom']].set_color(cmap(norm(ii)))
                ax.spines[['right','left','top','bottom']].set_linewidth(4)
                ax.set_yticks([])
                ax.set_xticks([])
        
            fig.suptitle(f"Z coordinate: {zz[0,0,z_index]:.3f}")
            if show:
                plt.show()
            else:
                plt.savefig(fn + f'_slice_{z_index+1}.png')
            plt.close()
