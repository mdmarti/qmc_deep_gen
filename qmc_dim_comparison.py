import torch
import os
from data.utils import load_data
from models.sampling import gen_fib_basis,fib,gen_korobov_basis
from models.utils import *
import train.train as train_qmc 
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
import mocap


def run_qmc_vae_experiments(save_location,dataloc,dataset,batch_size=256,nEpochs=300,rerun=False,train_lattice_m=15,make_comparison_plots=True,frames_per_sample=1):

    save_location = os.path.join(save_location,dataset)
    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    print(f"Training on {dataset} data")
    train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size,frames_per_sample=frames_per_sample)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    

    qmc_latent_dims = [1,2,3]


    data_save_loc = os.path.join(save_location,f'qmc_dim_comparison_stats_{dataset}.json')
    qmc_grid_loc = os.path.join(save_location,f'qmc_{dataset}_grid.png')
    for qmc_latent_dim in qmc_latent_dims:
        if not os.path.isfile(data_save_loc) and not rerun: 

            if qmc_latent_dim == 1:

                train_lattice = torch.linspace(0,1,fib(train_lattice_m))[:,None]
                test_lattice = torch.linspace(0,1,fib(20))[:,None]


            elif qmc_latent_dim == 2:

                train_lattice = gen_fib_basis(m=train_lattice_m)
                test_lattice = gen_fib_basis(m=20)
            else:
                train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size//2,frames_per_sample=frames_per_sample)

                train_lattice = gen_korobov_basis(a=76,num_dims=qmc_latent_dim,num_points=1021)
                test_lattice = gen_korobov_basis(a=1516,num_dims=qmc_latent_dim,num_points=4093)
            ############## QMC Training ########################

            qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim,n_per_sample=frames_per_sample)
            qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

            qmc_loss_func = binary_evidence if 'mnist' in dataset.lower() else lambda samples,data: gaussian_evidence(samples,data,var=0.1)
            qmc_lp = binary_lp if 'mnist' in dataset.lower() else lambda samples,data: gaussian_lp(samples,data,var=0.1)
            qmc_save_path = os.path.join(save_location,f'qmc_train_{qmc_latent_dim}d_{dataset}_dim_comparison.tar')

            if not os.path.isfile(qmc_save_path):
                qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop(qmc_model,train_loader,train_lattice.to(device),qmc_loss_func,\
                                                                    nEpochs=nEpochs,print_losses=dataset.lower() == 'celeba')
                print("Done training!")
                qmc_model.eval()
                with torch.no_grad():
                    qmc_test_losses = train_qmc.test_epoch(qmc_model,test_loader,test_lattice.to(device),qmc_loss_func)
                qmc_run_info = {'train':qmc_losses,'test':qmc_test_losses}
                save(qmc_model.to('cpu'),qmc_opt,qmc_run_info,fn=qmc_save_path)
                qmc_model.to(device)
                

            else:
                qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
                qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_save_path)
                qmc_losses,qmc_test_losses = qmc_run_info['train'],qmc_run_info['test']
                qmc_model.to(device)
                qmc_model.eval()
            print("making train plot...")
            vis2d.qmc_train_plot(qmc_losses,qmc_test_losses,save_fn=os.path.join(save_location,f'qmc_{qmc_latent_dim}d_{dataset}_train_curve.svg'))
            print("done!")
            if not os.path.isfile(qmc_grid_loc) and (dataset.lower() != 'mocap'):
                print("making model grid plot....")
                if qmc_latent_dim == 2:
                    vis2d.model_grid_plot(qmc_model,n_samples_dim=20,fn=qmc_grid_loc,
                                origin='lower' if dataset.lower() == 'finch' else None,
                                cm = 'viridis' if dataset.lower() == 'finch' else 'gray',
                                model_type='qmc',show=False)
                elif qmc_latent_dim == 1:
                    vis1d.model_grid_plot(qmc_model,n_samples_dim=20,fn=qmc_grid_loc,
                                        origin='lower' if dataset.lower() == 'finch' else None,
                                        cm = 'viridis' if dataset.lower() == 'finch' else 'gray',
                                        model_type='qmc',show=False)
                elif qmc_latent_dim == 3:
                    qmc_grid_loc = os.path.join(save_location,f'qmc_{dataset}_grid')
                    vis3d.model_grid_plot(qmc_model,n_samples_dim=20,fn=qmc_grid_loc,
                                          origin='lower' if dataset.lower() == 'finch' else None,
                                        cm = 'viridis' if dataset.lower() == 'finch' else 'gray',
                                        model_type='qmc',show=False)
                print("done!")
            elif not os.path.isfile(qmc_grid_loc) and (qmc_latent_dim == 2):

                mocap.model_grid_plot(qmc_model,n_samples_dim=20,base_motion=test_loader.dataset.motions[0][0],
                                      joints = test_loader.dataset.joints,conversion_key=test_loader.dataset.conversion_keys[0],
                                      fn = qmc_grid_loc,show=False,model_type='qmc')
            qmc_test_losses = -np.array(qmc_test_losses)


if __name__ =='__main__':

    fire.Fire(run_qmc_vae_experiments)
