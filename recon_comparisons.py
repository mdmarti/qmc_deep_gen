import torch
import plotting.visualize as vis2d
from train.model_saving_loading import load
from data.utils import load_data
from models.qmc_base import *
from models.vae_base import *
from models.sampling import *
from models.utils import *
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from plotting.visualize import format_plot_axis
from torch.optim import Adam
from train.losses import *
import train.train as train_qmc
from train import train_vae
import seaborn as sns
import plotting.visualize as vis2d


def get_dataloc(dataset):

    if 'gerbil' in dataset:
        return '/mnt/home/mmartinez/ceph/data/gerbil/ralph'
    if 'finch' in dataset:
        return '/mnt/home/mmartinez/ceph/bird_data/goffinet_21/blu285'
    if 'mocap' in dataset:
        return '/mnt/home/mmartinez/ceph/data/cmu_mocap/all_asfamc/subjects'
    if 'shapes3d' in dataset:
        return '/mnt/home/mmartinez/ceph/data/shapes3d'
    if 'mnist' in dataset:
        return '/mnt/home/mmartinez/ceph/data'
    if 'celeba' in dataset:
        return '/mnt/home/zkadkhodaie/ceph/datasets/img_align_celeba'
    
def run_reconstructions(n_per_sample=10):
    datasets = ['gerbil_ava_binom_upsample_fam245_full', #gerbil
                'mnist_simple', # mnist
                'celeba_scaled_res_layernorm', # celebA
                'finch', # finch
                'mocap_simple_10framespersample', # mocap
                'shapes3d_2d_layernorm']# 3dshapes
    var = 0.1
    plots_dir = '/mnt/home/mmartinez/ceph/qmc_experiments/fig2'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    lattice =gen_fib_basis(m=20)
    for dataset in datasets:
        print(f"now running for dataset {dataset}")
        if 'finch' in dataset.lower():
            cm = 'viridis'
            origin = 'lower'
        elif 'gerbil' in dataset.lower():
            cm = 'inferno'
            origin = 'lower'
        else:
            cm = 'gray'
            origin = None
        dataloc = get_dataloc(dataset)
        train_loader,test_loader = load_data(dataset,dataloc,batch_size=1,
                                     frames_per_sample=10,
                                             families=[2,4,5])
        base_path = '/mnt/home/mmartinez/ceph/qmc_experiments/qmc_vae_comparison'
        models_path = os.path.join(base_path,dataset)



        vae_model_file = os.path.join(models_path,f'vae_train_{dataset}_dim_comparison_2d.tar')
        qmc_model_file = os.path.join(models_path,f'qmc_train_{dataset}_2_dim_comparison.tar')
        if not os.path.isfile(qmc_model_file):
            qmc_model_file = os.path.join(models_path,f'qmc_train_{dataset}_dim_comparison.tar')
        iwae_model_file = os.path.join(models_path,f'iwae_train_{dataset}_dim_comparison_2d.tar')

        ##### set up iwae model ######
        iwae_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=2,arch='vae',n_per_sample=n_per_sample)
        iwae_encoder = get_encoder_arch(dataset_name=dataset,latent_dim=2,n_per_sample=n_per_sample)

        iwae_model = IWAE(decoder=iwae_decoder,encoder=iwae_encoder,
                        distribution=LowRankMultivariateNormal,device=device,k_samples=10)
        iwae_opt = Adam(iwae_model.parameters(),lr=1e-3)
        iwae_model,iwae_opt,iwae_run_info = load(iwae_model,iwae_opt,iwae_model_file)

        iwae_model.to(device)
        iwae_model.eval()
        ##### set up VAE model #########
        
        vae_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=2,arch='vae',n_per_sample=n_per_sample)
        vae_encoder = get_encoder_arch(dataset_name=dataset,latent_dim=2,n_per_sample=n_per_sample)

        vae_model = VAE(decoder=vae_decoder,encoder=vae_encoder,
                        distribution=LowRankMultivariateNormal,device=device)
        vae_opt = Adam(iwae_model.parameters(),lr=1e-3)
        vae_model,iwae_opt,iwae_run_info = load(iwae_model,iwae_opt,vae_model_file)

        vae_model.to(device)
        vae_model.eval()
        
        ##### set up QMC model ##########

        qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=2,arch='qmc',n_per_sample=n_per_sample)
        qmc_model = QMCLVM(latent_dim=2,device=device,decoder=qmc_decoder)

        qmc_lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
        qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
        qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_model_file)

        qmc_model.eval()
        qmc_model.to(device)
        ################################

        ############ Make plots ############

        plot_save_file_vae = os.path.join(plots_dir,f"{dataset}_vae_qmc_recons.svg")
        plot_save_file_iwae = os.path.join(plots_dir,f"{dataset}_iwae_qmc_recons.svg")
        plot_save_file_samples = os.path.join(plots_dir,f"{dataset}_iwae_vae_qmc_samples.svg")

        vis2d.recon_comparison_plot(qmc_model,qmc_lp,vae_model,
                                    test_loader,lattice,
                                    n_samples_comparison=8,
                          save_path=plot_save_file_vae,
                          cm=cm,origin=origin,show=False,
                          recon_type='rqmc',n_samples_recon=5)
        vis2d.recon_comparison_plot(qmc_model,qmc_lp,iwae_model,
                                    test_loader,lattice,
                                    n_samples_comparison=8,
                          save_path=plot_save_file_iwae,
                          cm=cm,origin=origin,show=False,
                          recon_type='rqmc',n_samples_recon=5)
        vis2d.sample_comparison_plot(qmc_model,iwae_model,
                                    vae_model,
                                    n_samples_comparison=8,
                          save_path=plot_save_file_samples,
                          cm=cm,origin=origin,show=False)

        

if __name__ == '__main__':

    run_reconstructions()