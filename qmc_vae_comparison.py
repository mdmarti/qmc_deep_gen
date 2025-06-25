import torch
import os
from data.utils import load_data
from models.sampling import gen_fib_basis
from models.utils import *
import train.train as train_qmc 
import train.train_vae as train_vae
from models.vae_base import VAE 
from models.qmc_base import QMCLVM
from train.losses import *
from train.model_saving_loading import *
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.optim import Adam
from plotting.visualize import recon_comparison_plot,posterior_comparison_plot
import matplotlib.pyplot as plt
import fire


def run_qmc_vae_experiments(save_location,dataloc,dataset,nEpochs=300):



    ################ Shared Setup ######################################
    #n_workers = len(os.sched_getaffinity(0))

    train_loader,test_loader = load_data(dataset,dataloc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_lattice = gen_fib_basis(m=16)
    test_lattice = gen_fib_basis(m=20)

    qmc_latent_dim=2

    vae_latent_dim = [qmc_latent_dim**ii for ii in range(1,10)]

    ############## QMC Training ########################

    qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim)
    qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

    qmc_loss_func = binary_evidence if dataset.lower() == 'mnist' else lambda samples,data: gaussian_evidence(samples,data,var=0.1)
    qmc_lp = binary_lp if dataset.lower() == 'mnist' else lambda samples,data: gaussian_lp(samples,data,var=0.1)
    qmc_save_path = os.path.join(save_location,f'qmc_train_{dataset}_dim_comparison.tar')

    if not os.path.isfile(qmc_save_path):
        qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop(qmc_model,train_loader,train_lattice.to(device),qmc_loss_func,\
                                                            nEpochs=nEpochs,print_losses=dataset.lower() == 'celeba')
        qmc_test_losses = train_qmc.test_epoch(qmc_model,test_loader,test_lattice.to(device),qmc_loss_func)
        qmc_run_info = {'train':qmc_losses,'test':qmc_test_losses}
        save(qmc_model.to('cpu'),qmc_opt,qmc_run_info,fn=qmc_save_path)
        qmc_model.to(device)
        

    else:
        qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
        qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_save_path)
        qmc_losses,qmc_test_losses = qmc_run_info['train'],qmc_run_info['test']
        qmc_model.to(device)
    
    qmc_test_losses = -np.array(qmc_test_losses)

    ############## VAE Training ######################

    vae_test_recons_all,vae_test_kls_all = [],[]
    vae_loss_func = binary_elbo if dataset.lower() == 'mnist' else lambda recons,distribution,data: gaussian_elbo(recons,distribution,data,recon_precision=10)
    vae_lp = binary_lp if dataset.lower() == 'mnist' else lambda target,recon: gaussian_lp(recon,target,var=0.1)
    for ld in vae_latent_dim:

        vae_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=ld,arch='vae')
        vae_encoder = get_encoder_arch(dataset_name=dataset,latent_dim=ld)

        vae_model = VAE(decoder=vae_decoder,encoder=vae_encoder,
                        distribution=LowRankMultivariateNormal,device=device)
        
        vae_save_path = os.path.join(save_location,f'vae_train_{dataset}_dim_comparison_{ld}d.tar')

        if not os.path.isfile(vae_save_path):
            print(f"now training VAE with latent dim {ld}")

            vae_model,vae_opt,vae_losses = train_vae.train_loop(vae_model,train_loader,vae_loss_func,nEpochs=nEpochs)
            vae_test_losses = train_vae.test_epoch(vae_model,test_loader,vae_loss_func)
            vae_run_info = {'train':vae_losses,'test':vae_test_losses}
            save(vae_model.to('cpu'),vae_opt,vae_run_info,fn=vae_save_path)
            vae_model.to(device)

        else:
            print(f"Now loading VAE with latent dim {ld}")
            vae_opt = Adam(vae_model.parameters(),lr=1e-3)
            vae_model,vae_opt,vae_run_info = load(vae_model,vae_opt,vae_save_path)
            vae_losses,vae_test_losses = vae_run_info['train'],vae_run_info['test']
            vae_model.to(device)

        [vae_test_recons,vae_test_kls] = vae_test_losses
        vae_test_recons,vae_test_kls = -np.array(vae_test_recons),np.array(vae_test_kls)
        vae_test_recons_all.append(vae_test_recons)
        vae_test_kls_all.append(vae_test_kls)
        recon_save_loc = os.path.join(save_location,"qmc_vae_recon_comparison_" + str(ld) + 'd_{sample_num}.png')
        recon_comparison_plot(qmc_model,qmc_lp,vae_model,test_loader,test_lattice.to(device),n_samples=50,save_path=recon_save_loc)

        if ld == 2:
            posterior_save_loc =os.path.join(save_location,"vae_posterior_comparison_" + str(ld) + 'd_{sample_num}.png')
            posterior_comparison_plot(vae_model,test_loader,vae_lp,save_path=posterior_save_loc)




    qmc_mu_ev, qmc_sd_ev = np.nanmean(qmc_test_losses),np.nanstd(qmc_test_losses)
    vae_mu_recons = np.array([np.nanmean(r) for r in vae_test_recons_all])
    vae_sd_recons = np.array([np.nanstd(r) for r in vae_test_recons_all])
    vae_mu_elbos = np.array([np.nanmean(r - k) for r,k in zip(vae_test_recons_all,vae_test_kls_all)])
    vae_sd_elbos = np.array([np.nanstd(r - k) for r,k in zip(vae_test_recons_all,vae_test_kls_all)])


    ax = plt.gca()
    ax.errorbar(vae_latent_dim,vae_mu_recons,yerr = vae_sd_recons,capsize=12,color='tab:green',fmt='o')
    ax.errorbar(vae_latent_dim+.1,vae_mu_elbos,yerr = vae_sd_elbos,capsize=12,color='tab:orange',fmt='o')
    ax.hlines(qmc_mu_ev,xmin=0,xmax=128,color='k')
    ax.hlines([qmc_mu_ev + qmc_sd_ev,qmc_mu_ev - qmc_sd_ev],xmin=0,xmax=128,color='k',linestyle='--')
    ax.set_xlim((0,128))
    ax.set_ylim(-300,0)
    plt.show()
    plt.close()


if __name__ =='__main__':

    fire.Fire(run_qmc_vae_experiments)

        





