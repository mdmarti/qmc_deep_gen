import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import scipy.stats as stats #Normal

import torch
import gc

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
        xx,yy = torch.meshgrid([torch.linspace(EPS1,1-EPS2,n_samples_dim)]*2,indexing='ij')
        z = torch.stack([xx.flatten(),yy.flatten()],axis=-1).to(device)
        if model_type == 'vae':

            dist = stats.norm
            z = torch.from_numpy(dist.ppf(z.detach().cpu().numpy())).to(torch.float32).to(model.device)
        sample = model.decoder(z).detach().cpu()
    z = z.detach().cpu().numpy()
    inds = np.arange(n_samples)
    cs = cmap(norm(inds))

    
    mosaic = [[f"sample {ii*n_samples_dim + jj}" for ii in range(n_samples_dim)] for jj in range(n_samples_dim)]                
    

    fig, axes = plt.subplot_mosaic(mosaic,figsize=(20,20),sharex=True,sharey=True,gridspec_kw={'wspace':0.01,'hspace':0.01})

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

def vae_train_plot(vae_train_losses,vae_test_losses,save_fn):

    [vae_train_recons,vae_train_kls] = vae_train_losses
    vae_train_recons,vae_train_kls = np.array(vae_train_recons),np.array(vae_train_kls)
    [vae_test_recons,vae_test_kls] = vae_test_losses
    vae_test_recons,vae_test_kls = np.array(vae_test_recons),np.array(vae_test_kls)

    N = len(vae_train_recons)
    ax = plt.gca()
    l1,=ax.plot(vae_train_kls,label='VAE KL',color='tab:orange')
    l2,=ax.plot(-vae_train_recons,label='VAE reconstruction log probability',color='tab:blue')
    l3,=ax.plot(-vae_train_recons - vae_train_kls,label='VAE ELBO',color='tab:green')

    xax = list(ax.get_xticks()) 
    xticklabels = xax + ['Test']
    xax += [N + N/100] 

    ax.errorbar(N + N/100,np.nanmean(-vae_test_recons),yerr  = np.nanstd(-vae_test_recons),color='tab:blue',capsize=6,linestyle='')
    ax.errorbar(N + N/100,np.nanmean(-vae_test_kls),yerr  = np.nanstd(-vae_test_kls),color='tab:orange',capsize=6,linestyle='')
    ax.errorbar(N + N/100,np.nanmean(-vae_test_recons - vae_test_kls),yerr  = np.nanstd(-vae_test_recons - vae_test_kls),color='tab:green',capsize=6,linestyle='')
    ax =  format_plot_axis(ax,ylabel='',xlabel='update number',xticks=xax,xticklabels=xticklabels)
    ax.legend([l1,l2,l3],['KL','log likelihood','ELBO'],frameon=False)
    plt.savefig(save_fn)
    plt.close()


def qmc_train_plot(qmc_train_losses,qmc_test_losses,save_fn):

    qmc_train_losses,qmc_test_losses = np.array(qmc_train_losses),np.array(qmc_test_losses)

    ax = plt.gca()
    N = len(qmc_train_losses)

    ax.plot(-qmc_train_losses,label = 'Model evidence',color='tab:blue')
    xax = list(ax.get_xticks()) 
    xticklabels = xax + ['Test']
    xax += [N + N//10] 
    ax.errorbar(N + N//10,np.nanmean(-qmc_test_losses),yerr =np.nanstd(-qmc_test_losses),capsize=6,linestyle='',color='tab:blue')
    ax =  format_plot_axis(ax,ylabel='Model evidence',xlabel='update number',xticks=xax,xticklabels=xticklabels)
    plt.savefig(save_fn)
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

def round_trip_qmc(model,grid,data,log_density,device='cuda'):
    #grid = grid.to(device)
    ### make this a model method

    posterior = model.posterior_probability(grid.to(device),data.to(device),log_density).detach().cpu()
    data = data.detach().cpu()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    #+preds = model.decoder(grid.to(device)).detach().cpu()
    #print(posterior.shape)
    #print(preds.shape)
    assert torch.sum(posterior) == 1, print(torch.sum(posterior))
    most_likely_grid = (grid * posterior[:,None]).sum(dim=0)

    recon = model(most_likely_grid,random=False)

    return data,recon

def recon_comparison_plot(qmc_model,qmc_likelihood,vae_model,loader,qmc_lattice,n_samples=50,
                          save_path='recon_{sample_num}.png',cm='gray',origin=None,show=False):

    n_samples = min(n_samples,len(loader.dataset))
    sample_inds = np.random.choice(len(loader.dataset),n_samples,replace=False).squeeze()
    
    for sample_ind in tqdm(sample_inds):

        save_path_ind = save_path.format(sample_num = sample_ind)
        sample = loader.dataset[sample_ind][0].to(torch.float32).to(qmc_model.device)
        sample = sample.view(1,1,sample.shape[-2],sample.shape[-1])
        recon_qmc = qmc_model.round_trip(qmc_lattice,sample,qmc_likelihood,recon_type='posterior').detach().cpu()
        recon_vae = vae_model.round_trip(sample).detach().cpu()
        sample = sample.detach().cpu().numpy()

        fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(10,5),sharex=True,sharey=True)
        axs[0].imshow(recon_qmc.squeeze(),cmap=cm,origin=origin)
        #axs[1].imshow(recon_qmc2.squeeze(),cmap='gray')
        axs[1].imshow(sample.squeeze(),cmap=cm,origin=origin)
        axs[2].imshow(recon_vae.squeeze(),cmap=cm,origin=origin)
        labels=['QMC reconstruction','Original image','VAE reconstruction'] #'QMC max prob point',
        for ax,label in zip(axs,labels):
            ax = format_img_axis(ax,title=label)
            
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(save_path_ind)
        plt.close()

    return

def posterior_comparison_plot(vae_model,loader,log_prob,n_samples=20,n_points=50,show=False,save_path='posterior_{sample_num}.png'):

    n_samples = min(n_samples,len(loader.dataset))

    sample_inds = np.random.choice(len(loader.dataset),n_samples,replace=False)

    for sample_ind in tqdm(sample_inds):

        save_path_sample = save_path.format(sample_num=sample_ind)
        sample = loader.dataset[sample_ind][0]
        sample = sample.view(1,1,sample.shape[-2],sample.shape[-1]).to(torch.float32).to(vae_model.device)
        emp_posterior,enc_posterior,grid = vae_model.posterior(sample,n_points,log_prob)


        emp_weighted_grid = emp_posterior.T @ grid
        prop_weighted_grid = enc_posterior.T @ grid

        recon_post_emp = vae_model.decoder(torch.from_numpy(emp_weighted_grid).view(1,2).to(torch.float32).to(vae_model.device)).detach().cpu().numpy().squeeze()
        recon_post_enc = vae_model.decoder(torch.from_numpy(prop_weighted_grid).view(1,2).to(torch.float32).to(vae_model.device)).detach().cpu().numpy().squeeze()
        
        
        mosaic = [['.','Sample','Sample','.'],
            ['Post1','Post1','Post2','Post2'],
         ['Recon1','Recon1','Recon2','Recon2']]

        fig, axs = plt.subplot_mosaic(mosaic,figsize=(8,10))
        #fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
        axs['Sample'].imshow(sample.detach().cpu().numpy().squeeze(),cmap='gray')
        axs['Sample'] = format_img_axis(axs['Sample'],title="Original sample")
        axs['Post1'].scatter(grid[:,0],grid[:,1],c=emp_posterior)
        axs['Post2'].scatter(grid[:,0],grid[:,1],c=enc_posterior)
        axs['Post1'] = format_plot_axis(axs['Post1'],title="Empirical posterior\n(based on decoder)",xticks=axs['Post1'].get_xticks(),yticks=axs['Post1'].get_yticks())
        axs['Post2'] = format_plot_axis(axs['Post2'],title="Encoder proposal distribution",xticks=axs['Post2'].get_xticks(),yticks=axs['Post2'].get_yticks())

        axs['Recon1'].imshow(recon_post_emp,cmap='gray')
        axs['Recon1'] = format_img_axis(axs['Recon1'],title="Recon from\nempirical posterior")

        axs['Recon2'].imshow(recon_post_enc,cmap='gray')
        axs['Recon2'] = format_img_axis(axs['Recon2'],title="Recon from encoder")
        if show:
            plt.show()
        else:
            plt.savefig(save_path_sample)
        plt.close()




def format_img_axis(ax,xlabel='',ylabel='',title=''):

    ax.spines[['left','right','top','bottom']].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def format_plot_axis(ax,xlabel='',ylabel='',title='',xticks=[],yticks=[],xticklabels=[],yticklabels=[],xlim=(),ylim=()):

    ax.spines[['right','top']].set_visible(False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(xticklabels) > 0:
        ax.set_xticks(xticks,xticklabels)
    elif len(xticks) > 0:
        ax.set_xticks(xticks)
    if len(yticklabels) > 0:
        ax.set_yticks(yticks,yticklabels)
    elif len(yticks) > 0:
        ax.set_yticks(yticks)
    if len(xlim) == 2:
        ax.set_xlim(xlim)
    if len(ylim) == 2: 
        ax.set_ylim(ylim)


    return ax





    

