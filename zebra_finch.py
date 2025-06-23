import torch
from models.sampling import *
from models.qmc_base import *
from models.layers import *
from train.losses import gaussian_lp,gaussian_evidence,gaussian_elbo
from models.vae_base import *
import train.train_vae as train_vae
import train.train as train_qmc
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
from torch.optim import Adam
from train.model_saving_loading import *
from plotting.visualize import *
from data.bird_data import *
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

import fire



def run_zebrafinch_experiments(save_location,dataloc,train_grid_m=15,test_grid_m=20,n_recons=50,nEpochs=300):

    ############ Set up experiment parameters ##################

    (train_files,test_files),(train_ids,test_ids) = load_segmented_sylls(dataloc,sylls=['A','B','C','D','D2','E'])
    train_ds = bird_data(train_files,train_ids)
    test_ds = bird_data(test_files,test_ids)
    n_workers = len(os.sched_getaffinity(0))
    train_loader = DataLoader(train_ds,num_workers=n_workers,shuffle=True,batch_size=64)
    test_loader = DataLoader(test_ds,num_workers=n_workers,shuffle=False,batch_size=1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #####################################################

    ############## QMC training #########################
    qmc_latent_dim=2
    qmc_loss_function = lambda samples,data: gaussian_evidence(samples,data,var=.1)
    
    decoder_qmc = nn.Sequential(
            TorusBasis(),
            nn.Linear(2*qmc_latent_dim,2048),
            nn.Linear(2048, 64*8*8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #nn.Linear(64*7*7,32*14*14),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),#nn.Linear(32*14*14,1*28*28),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid(),
        )
    
    qmc_model = QMCLVM(latent_dim=qmc_latent_dim, device=device,decoder=decoder_qmc)
    train_base_sequence = gen_fib_basis(m=train_grid_m)
    test_base_sequence = gen_fib_basis(m=test_grid_m)

    save_qmc = os.path.join(save_location,'qmc_train_zebra_finch_experiment.tar')
    if not os.path.isfile(save_qmc):
        print("now training qmc model")
        ## starting this run takes a little while for some reason...
        qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop(qmc_model,train_loader,train_base_sequence.to(device),qmc_loss_function,nEpochs=nEpochs)
        save(qmc_model.to('cpu'),qmc_opt,qmc_losses,fn=save_qmc)
        qmc_model.to(device)
    else:
        qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
        qmc_model,qmc_opt,qmc_losses = load(qmc_model,qmc_opt,save_qmc)

    qmc_losses = np.array(qmc_losses)
    ax = plt.gca()
    ax.plot(-np.array(qmc_losses))
    ax =  format_plot_axis(ax,ylabel='log evidence',xlabel='update number',xticks=ax.get_xticks(),yticks=ax.get_yticks())
    plt.savefig(os.path.join(save_location,'qmc_train_stats.svg'))
    plt.close()

    model_grid_plot(qmc_model.to(device),n_samples_dim=20,show=False,fn=os.path.join(save_location,'qmc_grid.png'),
                    origin='lower',cm='viridis')

    ######################################################


    ############## VAE training ##########################

    vae_latent_dim=32
    decoder_vae = nn.Sequential(
            nn.Linear(vae_latent_dim,2048),
            nn.Linear(2048, 64*8*8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #nn.Linear(64*7*7,32*14*14),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),#nn.Linear(32*14*14,1*28*28),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid(),
        )
    encoder_vae = Encoder(latent_dim=vae_latent_dim)
    #encoder_net =nn.Sequential(nn.Conv2d(1,32,3,stride=2,padding=1),
    #                        nn.Tanh(),
    #                        nn.Conv2d(32,64,3,stride=2,padding=1),
    #                        nn.Tanh(),
    #                        nn.Flatten(start_dim=1,end_dim=-1),
    #                        nn.Linear(64*7*7,2048),
    #                        nn.Tanh())
    #mu_net = nn.Linear(2048,vae_latent_dim)
    #L_net = nn.Linear(2048,vae_latent_dim)
    #d_net = nn.Linear(2048,vae_latent_dim)
    #encoder_vae = Encoder(encoder_net,mu_net,L_net,d_net,vae_latent_dim)
    vae_model= VAE(decoder_vae,encoder_vae,LowRankMultivariateNormal,device)
    vae_model.device=device    

    vae_loss_function = lambda recon,dist,target: gaussian_elbo(recon,dist,target,recon_precision=1/.1)

    save_vae = os.path.join(save_location,'vae_train_zebrafinch_experiment.tar')
    if not os.path.isfile(save_vae):
        print("now training vae model")
        vae_model,vae_opt,vae_losses = train_vae.train_loop(vae_model,train_loader,vae_loss_function,nEpochs=nEpochs)
        save(vae_model.to('cpu'),vae_opt,vae_losses,fn=save_vae)
        vae_model.to(device)
    else:
        vae_opt = Adam(vae_model.parameters(),lr=1e-3)
        vae_model,vae_opt,vae_losses = load(vae_model,vae_opt,save_vae)

    [vae_recons,vae_kls] = vae_losses
    vae_recons,vae_kls = np.array(vae_recons),np.array(vae_kls)
    ax = plt.gca()
    ax.plot(-vae_recons,label='VAE reconstruction log probability')
    ax.plot(vae_kls,label='VAE KL')
    ax.plot(-vae_recons - vae_kls,label='VAE ELBO')
    ax =  format_plot_axis(ax,ylabel='',xlabel='update number',xticks=ax.get_xticks(),yticks=ax.get_yticks())
    ax.legend(frameon=False)
    plt.savefig(os.path.join(save_location,'vae_train_stats.svg'))
    plt.close()


    ######################################################


    ############### analysis and comparison ##############


    ax = plt.gca()
    ax.plot(-qmc_losses,label='QMC model evidence')
    ax.plot(-vae_recons,label='VAE reconstruction log probability')
    ax.plot(-vae_recons - vae_kls,label='VAE ELBO')
    ax =  format_plot_axis(ax,ylabel='',xlabel='update number',xticks=ax.get_xticks(),yticks=ax.get_yticks())
    ax.legend(frameon=False)
    plt.savefig(os.path.join(save_location,'qmc_vae_stats_comparison.svg'))
    plt.close()

    sample_inds = np.random.choice(len(test_loader.dataset),n_recons,replace=False)
    lp_fnc = lambda x,y: gaussian_lp(x,y,var=0.1)
    for ii in range(n_recons):


        sample_ind = sample_inds[ii]
        save_fig = os.path.join(save_location,f'qmc_vae_round_trips_sample_{sample_ind}.png')

        sample = test_loader.dataset[sample_ind][0].to(torch.float32).to(device).view(1,1,128,128)
        recon_qmc1 = qmc_model.round_trip(test_base_sequence.to(device),sample,lp_fnc)
        recon_vae = vae_model.round_trip(sample)
        recon_qmc1 = recon_qmc1.detach().cpu()
        #recon_qmc2 = recon_qmc2.detach().cpu()
        recon_vae = recon_vae.detach().cpu()
        sample = sample.detach().cpu().squeeze()

        fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(10,5),sharex=True,sharey=True)
        axs[0].imshow(recon_qmc1.squeeze(),cmap='gray')
        #axs[1].imshow(recon_qmc2.squeeze(),cmap='gray')
        axs[1].imshow(sample.squeeze(),cmap='gray')
        axs[2].imshow(recon_vae.squeeze(),cmap='gray')
        labels=['QMC reconstruction','Original image','VAE reconstruction'] #'QMC max prob point',
        for ax,label in zip(axs,labels):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label)
        plt.tight_layout()
        plt.savefig(save_fig)
        plt.close()

    ######################################################################################################


if __name__ == '__main__':

    fire.Fire(run_zebrafinch_experiments)

