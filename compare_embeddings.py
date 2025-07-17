import torch
from umap import umap
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

def compare_embeddings(qmc_save_loc,
                       vae_2d_save_loc,
                       vae_128d_save_loc,
                       dataset,
                       dataloc,
                       save_location,
                       batch_size=256,
                       lattice_m=15,
                       frames_per_sample=10
                       ):
    
    save_location = os.path.join(save_location,dataset)
    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qmc_latent_dim=3 if (('celeba' in dataset.lower()) or ('shapes3d' in dataset.lower())) else 2

    if qmc_latent_dim == 2:
        train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size,frames_per_sample=frames_per_sample)
        lattice = gen_fib_basis(m=lattice_m)
    else:
        train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size//2,frames_per_sample=frames_per_sample)

        lattice = gen_korobov_basis(a=1516,num_dims=qmc_latent_dim,num_points=4093)

    qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim,n_per_sample=frames_per_sample)
    qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

    qmc_loss_func = binary_evidence if 'mnist' in dataset.lower() else lambda samples,data: gaussian_evidence(samples,data,var=0.1)
    qmc_lp = binary_lp if 'mnist' in dataset.lower() else lambda samples,data: gaussian_lp(samples,data,var=0.1)
    qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
    qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_save_loc)
    qmc_losses,qmc_test_losses = qmc_run_info['train'],qmc_run_info['test']
    qmc_model.to(device)
    qmc_model.eval()

    test_latents_qmc,test_labels_qmc = qmc_model.embed_data(lattice,test_loader,qmc_lp)
    train_latents_qmc,train_labels_qmc = qmc_model.embed_data(lattice,train_loader,qmc_lp)
    qmc_model.to('cpu')

    vae_loss_func = binary_elbo if 'mnist' in dataset.lower() else lambda recons,distribution,data: gaussian_elbo(recons,distribution,data,recon_precision=10)
    vae_lp = binary_lp if 'mnist' in dataset.lower() else lambda target,recon: gaussian_lp(recon,target,var=0.1)

    vae_decoder_2d = get_decoder_arch(dataset_name=dataset,latent_dim=2,arch='vae',n_per_sample=frames_per_sample)
    vae_encoder_2d = get_encoder_arch(dataset_name=dataset,latent_dim=2,n_per_sample=frames_per_sample)

    vae_model_2d = VAE(decoder=vae_decoder_2d,encoder=vae_encoder_2d,
                    distribution=LowRankMultivariateNormal,device=device)
    
    vae_opt = Adam(vae_model_2d.parameters(),lr=1e-3)
    vae_model_2d,vae_opt,vae_run_info = load(vae_model_2d,vae_opt,vae_2d_save_loc)
    vae_losses,vae_test_losses = vae_run_info['train'],vae_run_info['test']
    vae_model_2d.to(device)
    vae_model_2d.eval()

    test_latents_vae_2d,test_labels_vae_2d = vae_model_2d.embed_data(lattice,test_loader,qmc_lp)
    train_latents_vae_2d,train_labels_vae_2d = vae_model_2d.embed_data(lattice,train_loader,qmc_lp)
    vae_model_2d.to('cpu')
    
    vae_decoder_128d = get_decoder_arch(dataset_name=dataset,latent_dim=128,arch='vae',n_per_sample=frames_per_sample)
    vae_encoder_128d = get_encoder_arch(dataset_name=dataset,latent_dim=128,n_per_sample=frames_per_sample)

    vae_model_128d = VAE(decoder=vae_decoder_128d,encoder=vae_encoder_128d,
                    distribution=LowRankMultivariateNormal,device=device)
    
    vae_opt = Adam(vae_model_128d.parameters(),lr=1e-3)
    vae_model_128d,vae_opt,vae_run_info = load(vae_model_128d,vae_opt,vae_128d_save_loc)
    vae_losses,vae_test_losses = vae_run_info['train'],vae_run_info['test']
    vae_model_128d.to(device)
    vae_model_128d.eval()

    test_latents_vae_128d,test_labels_vae_128d = vae_model_128d.embed_data(lattice,test_loader,qmc_lp)
    train_latents_vae_128d,train_labels_vae_128d = vae_model_128d.embed_data(lattice,train_loader,qmc_lp)
    vae_model_2d.to('cpu')