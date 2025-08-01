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
from data.toy_dsets import get_3d_shapes_fixed_factors,_FACTORS_IN_ORDER
from torch.utils.data import DataLoader
import fire
import json

def run_fixed_factor_experiments(save_location,dataloc,batch_size=64,nEpochs=10,rerun=False,train_lattice_m=15,qmc_dim=2):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_workers = len(os.sched_getaffinity(0))

    if qmc_dim == 2:
        train_lattice = gen_fib_basis(m=train_lattice_m)
        test_lattice = gen_fib_basis(m=20)
    else:

        train_lattice = gen_korobov_basis(a=76,num_dims=qmc_dim,num_points=1021)
        test_lattice = gen_korobov_basis(a=1487,num_dims=qmc_dim,num_points=2039)

    ## Wall hue, floor hue, object hue, scale
    factor_order = [1,0,2,3]

    ### Object, scale, orientation, object hue
    # factor_order= [4,3,5,2]
    #factor_names = [_FACTORS_IN_ORDER[f] for f in factor_order]
    print(f"we will{'' if rerun else ' not'} be rerunning model analysis")
    for fixed_inds in range(1,len(factor_order)+1):

        fixed_factors = factor_order[:fixed_inds]
        fixed_factors.sort()
        factor_names = [_FACTORS_IN_ORDER[f] for f in fixed_factors]
        print("now fixing: " + ', '.join(factor_names))

        save_string = '_'.join(factor_names)


        data_save_loc = os.path.join(save_location,f'qmc_fixed_factors_{save_string}_stats_.json')
        qmc_grid_loc = os.path.join(save_location,f'qmc_fixed_factors_{save_string}_grid.png')

        train_ds,test_ds = get_3d_shapes_fixed_factors(dpath=dataloc,seed=92,fixed_factors=fixed_factors)
        train_loader = DataLoader(train_ds,num_workers=n_workers,shuffle=True,batch_size=batch_size)
        test_loader = DataLoader(test_ds,num_workers=n_workers,shuffle=False,batch_size=max(1,batch_size//2))
        if not os.path.isfile(data_save_loc) or rerun: 
            qmc_decoder = get_decoder_arch(dataset_name='shapes3d',latent_dim=qmc_dim)
            qmc_model = QMCLVM(latent_dim=qmc_dim,device=device,decoder=qmc_decoder)

            qmc_loss_func = lambda samples,data: gaussian_evidence(samples,data,var=0.1)
            qmc_lp = lambda samples,data: gaussian_lp(samples,data,var=0.1)
            qmc_save_path = os.path.join(save_location,f'qmc_train_shapes3d_{save_string}_fixed.tar')

            if not os.path.isfile(qmc_save_path):
                qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop(qmc_model,train_loader,train_lattice.to(device),qmc_loss_func,\
                                                                    nEpochs=nEpochs*fixed_inds)
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
        vis2d.qmc_train_plot(qmc_losses,qmc_test_losses,save_fn=os.path.join(save_location,f'qmc_fixed_factors_{save_string}_train_curve.svg'))
        print("done!")
        if not os.path.isfile(qmc_grid_loc):
            print("making model grid plot....")
            if qmc_dim == 2:
                vis2d.model_grid_plot(qmc_model,n_samples_dim=20,fn=qmc_grid_loc,
                                origin=None,
                                cm ='gray',
                                model_type='qmc',show=False)
           
            elif qmc_dim == 3:
                qmc_grid_loc = os.path.join(save_location,f'qmc_fixed_factors_{save_string}_grid')
                vis3d.model_grid_plot(qmc_model,n_samples_dim=20,fn=qmc_grid_loc,
                                        origin= None,
                                    cm = 'gray',
                                    model_type='qmc',show=False)
            print("done!")

if __name__ == '__main__':

    fire.Fire(run_fixed_factor_experiments)

