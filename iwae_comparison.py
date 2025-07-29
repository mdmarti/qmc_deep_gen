import torch
import os
from data.utils import load_data
from models.sampling import gen_fib_basis, gen_korobov_basis
from models.utils import *
import train.train_vae as train_vae
from models.vae_base import IWAE 
from train.losses import *
from train.model_saving_loading import *
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.optim import Adam
import plotting.visualize as vis2d # recon_comparison_plot,posterior_comparison_plot,qmc_train_plot,vae_train_plot,model_grid_plot
import matplotlib.pyplot as plt
import fire
import json


def run_iwae_experiments(save_location,dataloc,dataset,batch_size=256,
                            nEpochs=300,rerun=False,k_samples=10,
                            make_comparison_plots=True,frames_per_sample=1,
                            var=0.1,families=[2]):



    ################ Shared Setup ######################################
    #n_workers = len(os.sched_getaffinity(0))

    save_location = os.path.join(save_location,dataset)
    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    print(f"Training on {dataset} data")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'finch' in dataset.lower():
        cm = 'viridis'
        origin = 'lower'
    elif 'gerbil' in dataset.lower():
        cm = 'inferno'
        origin = 'lower'
    else:
        cm = 'gray'
        origin = None

    train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size,frames_per_sample=frames_per_sample,
                                             families=families)

    iwae_latent_dim = [2**ii for ii in range(1,8)]

    data_save_loc = os.path.join(save_location,f'iwae_dim_comparison_stats_{dataset}.json')
    print(f"we will{'' if rerun else ' not'} be rerunning model analysis")
    if not os.path.isfile(data_save_loc) or rerun: 

        ############## IWAE Training ######################
        ### recreating loader with larger batch size, so we can train everything faster
        #train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size*8,frames_per_sample=frames_per_sample,
        #                                     families=families)

        iwae_test_recons_all,iwae_test_kls_all = [],[]
        iwae_loss_func = binary_iwae_elbo if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda recons,distribution,data: gaussian_iwae_elbo(recons,distribution,data,recon_precision=1/var)
        iwae_lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda target,recon: gaussian_lp(recon,target,var=var)
        for ld in iwae_latent_dim:

            iwae_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=ld,arch='vae',n_per_sample=frames_per_sample)
            iwae_encoder = get_encoder_arch(dataset_name=dataset,latent_dim=ld,n_per_sample=frames_per_sample)

            iwae_model = IWAE(decoder=iwae_decoder,encoder=iwae_encoder,
                            distribution=LowRankMultivariateNormal,device=device,k_samples=k_samples)
            
            iwae_save_path = os.path.join(save_location,f'iwae_train_{dataset}_dim_comparison_{ld}d.tar')
            iwae_grid_loc = os.path.join(save_location,f'iwae_{dataset}_{ld}d_grid.png')

            if not os.path.isfile(iwae_save_path):
                print(f"now training IWAE with latent dim {ld}")

                iwae_model,iwae_opt,iwae_losses = train_vae.train_loop(iwae_model,train_loader,iwae_loss_func,nEpochs=nEpochs)
                iwae_model.eval()
                with torch.no_grad():
                    iwae_test_losses = train_vae.test_epoch(iwae_model,test_loader,iwae_loss_func)
                iwae_run_info = {'train':iwae_losses,'test':iwae_test_losses}
                save(iwae_model.to('cpu'),iwae_opt,iwae_run_info,fn=iwae_save_path)
                iwae_model.to(device)


            else:
                print(f"Now loading IWAE with latent dim {ld}")
                iwae_opt = Adam(iwae_model.parameters(),lr=1e-3)
                iwae_model,iwae_opt,iwae_run_info = load(iwae_model,iwae_opt,iwae_save_path)
                iwae_losses,iwae_test_losses = iwae_run_info['train'],iwae_run_info['test']
                iwae_model.to(device)
                iwae_model.eval()

            [iwae_test_recons,iwae_test_kls] = iwae_test_losses
            iwae_test_recons,iwae_test_kls = -np.array(iwae_test_recons),np.array(iwae_test_kls)
            iwae_test_recons_all.append(iwae_test_recons)
            iwae_test_kls_all.append(iwae_test_kls)
            print("making train plot...")
            vis2d.vae_train_plot(iwae_losses,iwae_test_losses,save_fn=os.path.join(save_location,f'iwae_{dataset}_{ld}d_train_curve.svg'))
            print("done!")
            if not os.path.isfile(iwae_grid_loc) and ld == 2 and (dataset.lower() != 'mocap'):
                print("making model grid plot....")
                vis2d.model_grid_plot(iwae_model,n_samples_dim=20,fn=iwae_grid_loc,
                                origin=origin,
                                cm = cm,
                                model_type='vae',show=False)
                print("done!")
            
            if ld == 2 and  ('mocap' not in dataset.lower()) and make_comparison_plots:
                print("comparing true to encoder posteriors (under decoder)")
                with torch.no_grad():
                    posterior_save_loc =os.path.join(save_location,"iwae_posterior_comparison_" + str(ld) + 'd_{sample_num}.png')
                    vis2d.posterior_comparison_plot(iwae_model,test_loader,iwae_lp,save_path=posterior_save_loc)
                print("Done!")


        
        iwae_mu_recons = np.array([np.nanmean(r) for r in iwae_test_recons_all])
        iwae_sd_recons = np.array([np.nanstd(r) for r in iwae_test_recons_all])
        iwae_mu_elbos = np.array([np.nanmean(r - k) for r,k in zip(iwae_test_recons_all,iwae_test_kls_all)])
        iwae_sd_elbos = np.array([np.nanstd(r - k) for r,k in zip(iwae_test_recons_all,iwae_test_kls_all)])

        
        plot_data = {'recon_mu': iwae_mu_recons.tolist(),
                    'recon_sd': iwae_sd_recons.tolist(),
                    'elbo_mu': iwae_mu_elbos.tolist(),
                    'elbo_sd': iwae_sd_elbos.tolist()
                    }
        with open(data_save_loc,'w') as f:
            json.dump(plot_data,f)
    else:
        with open(data_save_loc,'r') as f:

            plot_data = json.load(f)

        #qmc_mu_ev = np.array(plot_data['ev_mu'])#[:len(vae_latent_dim)]
        #qmc_sd_ev = np.array(plot_data['ev_sd'])#[:len(vae_latent_dim)]
        iwae_mu_recons = np.array(plot_data['recon_mu'])[:len(iwae_latent_dim)]
        iwae_sd_recons = np.array(plot_data['recon_sd'])[:len(iwae_latent_dim)]
        iwae_mu_elbos = np.array(plot_data['elbo_mu'])[:len(iwae_latent_dim)]
        iwae_sd_elbos = np.array(plot_data['elbo_sd'])[:len(iwae_latent_dim)]

    range_of_vals = [np.amin(iwae_mu_elbos - iwae_sd_elbos)  ,np.amax(iwae_mu_elbos + iwae_sd_elbos)]
    padding = (range_of_vals[1] - range_of_vals[0]) // 2

    ax = plt.gca()
    #ax.set_xscale('log')
    #r = ax.errorbar(iwae_latent_dim,iwae_mu_recons,yerr = iwae_sd_recons,capsize=12,color='tab:green',fmt='.')
    e= ax.errorbar(iwae_latent_dim,iwae_mu_elbos,yerr = iwae_sd_elbos,capsize=12,color='tab:orange',fmt='.')
    #q = ax.hlines(qmc_mu_ev,xmin=-20,xmax=iwae_latent_dim[-1]+20,color='k')

    #ax.hlines([qmc_mu_ev + qmc_sd_ev,qmc_mu_ev - qmc_sd_ev],xmin=-20,xmax=iwae_latent_dim[-1]+20,color='k',linestyle='--')

    ax.set_xlim((1,iwae_latent_dim[-1]+20))
    ax.set_ylim(range_of_vals[0] - padding,range_of_vals[1] + padding)
    ax.set_xlabel("IWAE latent dimension")
    #ax.set_xticks(vae_latent_dim)

    #
    ax.legend([q,r.lines[0],e.lines[0]],['IWAE ELBO'],frameon=False)
    plt.savefig(os.path.join(save_location,f'iwae_elbo_comparison_by_dim_{dataset}.svg'))
    #plt.show()
    plt.close()


if __name__ =='__main__':

    fire.Fire(run_iwae_experiments)

        





