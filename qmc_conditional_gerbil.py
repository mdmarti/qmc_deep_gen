import torch
import os
from data.utils import load_data
from models.sampling import gen_fib_basis, gen_korobov_basis
from models.utils import *
import train.train as train_qmc 
import train.train_vae as train_vae
from models.vae_base import VAE 
from models.qmc_base import QMCLVM
from train.losses import *
from train.model_saving_loading import *
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.optim import Adam
import plotting.visualize as vis2d # recon_comparison_plot,posterior_comparison_plot,qmc_train_plot,vae_train_plot,model_grid_plot
import plotting.visualize_1d as vis1d
import plotting.visualize_3d as vis3d
import matplotlib.pyplot as plt
import fire
import json

def run_conditioning_experiments(save_location,dataloc,nEpochs=100,
                                 batch_size=1,train_lattice_m=15,
                                 var=0.025,families=[2,4,5],conditional_factor='fm'):
    
    dataset ='conditional_gerbil_ava'
    save_location = os.path.join(save_location,dataset)
    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    print(f"Training on {dataset} data")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qmc_latent_dim=2
    cm = 'inferno'
    origin = 'lower'

    train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size,
                                             families=families,
                                             conditional=True,
                                             conditional_factor=conditional_factor)
    train_lattice = gen_fib_basis(m=train_lattice_m)
    test_lattice = gen_fib_basis(m=20)

    data_save_loc = os.path.join(save_location,f'qmc_conditional_{conditional_factor}_stats_{dataset}.json')
    qmc_grid_loc = os.path.join(save_location,f'qmc_{dataset}_grid_{qmc_latent_dim}d.png')

    if not os.path.isfile(data_save_loc):
        qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim,
                                arch = 'conditional_qmc')
        qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

        qmc_loss_func = lambda samples,data: gaussian_evidence(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
        qmc_lp =  lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
        qmc_save_path = os.path.join(save_location,f'qmc_train_{dataset}_conditioned_{conditional_factor}.tar')
        if not os.path.sifile(qmc_save_path):
            qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop(qmc_model,train_loader,train_lattice.to(device),qmc_loss_func,\
                                                                    nEpochs=nEpochs,verbose=False,
                                                                    conditional=True)
            
            print("Done training!")
            qmc_model.eval()
            with torch.no_grad():
                qmc_model.eval()
                qmc_test_losses = train_qmc.test_epoch(qmc_model,test_loader,test_lattice.to(device),qmc_loss_func)
            qmc_run_info = {'train':qmc_losses,'test':qmc_test_losses}
            save(qmc_model.to('cpu'),qmc_opt,qmc_run_info,fn=qmc_save_path)
            qmc_model.to(device)
            

        else:
            qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
            qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_save_path)
            qmc_model.eval()
            qmc_losses,qmc_test_losses = qmc_run_info['train'],qmc_run_info['test']
            qmc_model.to(device)
            qmc_model.eval()
        print("making train plot...")
        vis2d.qmc_train_plot(qmc_losses,qmc_test_losses,save_fn=os.path.join(save_location,f'qmc_{dataset}_train_curve.svg'))
        print("done!")
        qmc_test_losses = -np.array(qmc_test_losses)
        qmc_mu_ev, qmc_sd_ev = np.nanmean(qmc_test_losses),np.nanstd(qmc_test_losses)
        plot_data = {'ev_mu':qmc_mu_ev.tolist(),
                    'ev_sd':qmc_sd_ev.tolist()}   
        with open(data_save_loc,'w') as f:
                json.dump(plot_data,f)
    else:
        with open(data_save_loc,'r') as f:

            plot_data = json.load(f)

        qmc_mu_ev = np.array(plot_data['ev_mu'])#[:len(vae_latent_dim)]
        qmc_sd_ev = np.array(plot_data['ev_sd'])#[:len(vae_latent_dim)]    

    return 

if __name__ == '__main__':

    fire.Fire(run_conditioning_experiments)
