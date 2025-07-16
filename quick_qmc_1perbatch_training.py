import os
import glob
import numpy as np
#import third_party.amc_parser as amc
from data.mocap import *
from data.utils import *

import torch
import os
from data.utils import load_data
from models.sampling import gen_fib_basis,fib,gen_korobov_basis,gen_samples_batch
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
from tqdm import tqdm
import fire


def run_1_epoch(nperbatch=1):

    dataset = 'celeba_scaled_res'
    dataloc = '/mnt/home/zkadkhodaie/ceph/datasets/img_align_celeba'#'/mnt/home/mmartinez/ceph/data'
    save_location= f'/mnt/home/mmartinez/ceph/qmc_experiments/qmc_vae_comparison/{dataset}'
    qmc_latent_dim=3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader,test_loader = load_data(dataset,dataloc,batch_size=nperbatch)
    qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim)
    qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

    qmc_loss_func = binary_evidence if 'mnist' in dataset.lower() else lambda samples,data: gaussian_evidence(samples,data,var=0.1)
    qmc_lp = binary_lp if 'mnist' in dataset.lower() else lambda samples,data: gaussian_lp(samples,data,var=0.1)
    qmc_save_path = os.path.join(save_location,f'qmc_train_{dataset}_dim_comparison.tar')
    qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
    qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_save_path)
    qmc_losses,qmc_test_losses = qmc_run_info['train'],qmc_run_info['test']


    train_lattice = gen_korobov_basis(a=1487,num_dims=qmc_latent_dim,num_points=2093)
    test_lattice = gen_korobov_basis(a=1516,num_dims=qmc_latent_dim,num_points=4093)
    #train_lattice,test_lattice = gen_fib_basis(m=15), gen_fib_basis(m=20)

    new_train_losses= []
    qmc_model.train()
    for batch in tqdm(train_loader,desc="one train epoch",total = len(train_loader)):
        (data,label) = batch
        data = data.to(torch.float32).to(qmc_model.device)
        qmc_opt.zero_grad()
        s,w = gen_samples_batch(test_lattice,qmc_model,qmc_lp,data,2093)
        preds = qmc_model(s.to(qmc_model.device))
        l = gaussian_evidence(preds,data,var=0.1,importance_weights=w)
        l.backward()
        qmc_opt.step()
        new_train_losses.append(l.item())
    qmc_model.eval()

    new_test_losses=[]
    for batch in tqdm(test_loader,desc="one test epoch",total=len(test_loader)):
        (data,label) = batch
        data = data.to(torch.float32).to(qmc_model.device)
        with torch.no_grad():
            s,w = gen_samples_batch(test_lattice,qmc_model,qmc_lp,data,2093)
            preds = qmc_model(s.to(qmc_model.device))
            l = gaussian_evidence(preds,data,var=0.1,importance_weights=w)
            new_test_losses.append(l.item())


    nTrain1 = len(qmc_losses)
    nTrain2 = len(new_train_losses)

    xax1 = np.arange(0,nTrain1)
    xax2 = np.arange(nTrain1,nTrain1 + nTrain2)

    ax = plt.gca()
    ax.plot(xax1,qmc_losses,color='tab:blue',label="Old training")
    ax.plot(xax2,np.array(new_train_losses),color='tab:orange',label="New training")
    ax.set_xlabel("Training updates")
    ax.set_ylabel("Negative (log) evidence")
    ax.spines[['right','top']].set_visible(False)
    plt.savefig(f"/mnt/home/mmartinez/ceph/qmc_experiments/extra_training/train2_loss_{nperbatch}_per_batch.svg")
    plt.close()

    ax = plt.gca()
    qmc_mu_ev, qmc_sd_ev = np.nanmean(qmc_test_losses),np.nanstd(qmc_test_losses)
    qmc_mu_ev2, qmc_sd_ev2 = np.nanmean(new_test_losses),np.nanstd(new_test_losses)

    ax.errorbar([1],qmc_mu_ev,yerr=qmc_sd_ev,color='tab:blue',capsize=12,linestyle='.')
    ax.errorbar([2],qmc_mu_ev2,yerr=qmc_sd_ev2,color='tab:orange',capsize=12,linestyle='.')
    ax.set_xticks([1,2],['Old test','New test'])
    ax.spines[['right','top']].set_visible(False)
    plt.savefig(f"/mnt/home/mmartinez/ceph/qmc_experiments/extra_training/test2_loss_{nperbatch}_per_batch.svg")
    plt.close()

if __name__ == '__main__':

    fire.Fire(run_1_epoch)    
