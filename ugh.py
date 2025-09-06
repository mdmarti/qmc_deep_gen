


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
    
def run_annoying_test_samples():
    datasets = ['gerbil_ava_binom_upsample_fam245', #gerbil
                'mnist_simple', # mnist
                'celeba_scaled_res_layernorm', # celebA
                'finch', # finch
                'mocap_simple_10framespersample', # mocap
                'shapes3d_2d_layernorm']# 3dshapes
    for dataset in datasets:
        print(f"now running for dataset {dataset}")
        dataloc = get_dataloc(dataset)

        base_path = '/mnt/home/mmartinez/ceph/qmc_experiments/qmc_vae_comparison'
        stats_path = os.path.join(base_path,dataset)
        plots_dir = '/mnt/home/mmartinez/ceph/qmc_experiments/fig2'
        plot_save_file = os.path.join(plots_dir,f"{dataset}_evidence.svg")
        assert os.path.isdir(base_path)
        assert os.path.isdir(stats_path)
        
        vae_stats = os.path.join(stats_path,f'vae_qmc_dim_comparison_stats_{dataset}.json')
        iwae_stats = os.path.join(stats_path,f'iwae_dim_comparison_stats_{dataset}.json')
        all_stats_persample = os.path.join(stats_path,f'comparison_test_rerun.json')
        all_stats_persample = os.path.join(stats_path,f'comparison_test_rerun.json')

        train_loader,test_loader = load_data(dataset,dataloc,batch_size=1,
                                     frames_per_sample=10,
                                             families=[2,4,5])
        if os.path.isfile(all_stats_persample):

            print('loading stats')
            with open(all_stats_persample,'r') as f:

                plot_data = json.load(f)
        
            qmc_test_losses = np.array(plot_data['qmc'])
            test_losses_dims_vae = [np.array(l) for l in plot_data['vae']]
            test_losses_dims_iwae = [np.array(l) for l in plot_data['iwae']]
        else:
            print("sorry your file doesnt exist you'd better rerung")

            ############# QMC #
            qmc_latent_dim=2 #3 if (('celeba' in dataset.lower()) or ('shapes3d' in dataset.lower())) else 2
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            latent_dims = [2**ii for ii in range(1,8)]


            if qmc_latent_dim == 2:
                
                train_lattice = gen_fib_basis(m=15)
                test_lattice = gen_fib_basis(m=20)
            else:
            
                train_lattice = gen_korobov_basis(a=76,num_dims=qmc_latent_dim,num_points=1021)
                test_lattice = gen_korobov_basis(a=1487,num_dims=qmc_latent_dim,num_points=2039)


            qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim,n_per_sample=10)
            qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

            qmc_loss_func = binary_evidence if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda samples,data: gaussian_evidence(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
            qmc_lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
            
            qmc_save_path = os.path.join(stats_path,f'qmc_train_{dataset}_dim_comparison.tar')
            if not os.path.isfile(qmc_save_path):
                qmc_save_path = os.path.join(stats_path,f'qmc_train_{dataset}_2_dim_comparison.tar')

            qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
            qmc_model,qmc_opt,_ = load(qmc_model,qmc_opt,qmc_save_path) 

            qmc_model.eval()
            qmc_model.to(device)
            qmc_test_losses = -np.array(train_qmc.test_epoch(qmc_model,test_loader,test_lattice.to(device),
                                                qmc_loss_func))
            
            ########## VAE #

            test_losses_dims_vae = []
            var = 0.1
            vae_loss_func = binary_elbo if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda recons,distribution,data: gaussian_elbo(recons,distribution,data,recon_precision=1/var) #or ('gerbil' in dataset.lower()) 
            vae_lp = binary_lp if ('mnist' in dataset.lower())  or ('gerbil' in dataset.lower()) else lambda target,recon: gaussian_lp(recon,target,var=var) #or ('gerbil' in dataset.lower())

            for ld in latent_dims:
                vae_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=ld,arch='vae',n_per_sample=10)
                vae_encoder = get_encoder_arch(dataset_name=dataset,latent_dim=ld,n_per_sample=10)

                vae_model = VAE(decoder=vae_decoder,encoder=vae_encoder,
                                distribution=LowRankMultivariateNormal,device=device)
                
                vae_save_path = os.path.join(stats_path,f'vae_train_{dataset}_dim_comparison_{ld}d.tar')
                vae_opt = Adam(vae_model.parameters(),lr=1e-3)
                vae_model,vae_opt,_ = load(vae_model,vae_opt,vae_save_path)
                vae_model.eval()
                with torch.no_grad():
                    vae_test_losses = train_vae.test_epoch(vae_model,test_loader,vae_loss_func)
                [vae_test_recons,vae_test_kls] = vae_test_losses
                vae_test_recons,vae_test_kls = -np.array(vae_test_recons),np.array(vae_test_kls)
                test_losses_dims_vae.append(vae_test_recons - vae_test_kls)
                #print(test_losses_dims_vae[-1].shape)


            iwae_loss_func = binary_iwae_elbo if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda recons,distribution,data: gaussian_iwae_elbo(recons,distribution,data,recon_precision=1/var)
            iwae_lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda target,recon: gaussian_lp(recon,target,var=var)

            test_losses_dims_iwae = []
            for ld in latent_dims:
                iwae_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=ld,arch='vae',n_per_sample=10)
                iwae_encoder = get_encoder_arch(dataset_name=dataset,latent_dim=ld,n_per_sample=10)

                iwae_model = IWAE(decoder=iwae_decoder,encoder=iwae_encoder,
                                distribution=LowRankMultivariateNormal,device=device,k_samples=10)
                
                iwae_save_path = os.path.join(stats_path,f'iwae_train_{dataset}_dim_comparison_{ld}d.tar')

                iwae_opt = Adam(iwae_model.parameters(),lr=1e-3)
                iwae_model,iwae_opt,_ = load(iwae_model,iwae_opt,iwae_save_path)
                iwae_model.eval()
                with torch.no_grad():
                    iwae_test_losses = train_vae.test_epoch(iwae_model,test_loader,iwae_loss_func)
                [iwae_test_recons,iwae_test_kls] = iwae_test_losses
                iwae_test_recons,iwae_test_kls = -np.array(iwae_test_recons),np.array(iwae_test_kls)
                test_losses_dims_iwae.append(iwae_test_recons - iwae_test_kls)
                #print(test_losses_dims_iwae[-1].shape)

            plot_data = {
                    'qmc':qmc_test_losses.tolist(),
                    'vae':[l.tolist() for l in test_losses_dims_vae],
                    'iwae':[l.tolist() for l in test_losses_dims_iwae]}
            with open(all_stats_persample,'w') as f:
                json.dump(plot_data,f)

       
        ax = plt.gca()
        #ax.boxplot([qmc_test_losses,test_losses_dims_vae[0],test_losses_dims_iwae[0],test_losses_dims_vae[-1],test_losses_dims_iwae[-1]])
        sns.violinplot(data=[qmc_test_losses,test_losses_dims_vae[0],test_losses_dims_iwae[0],test_losses_dims_vae[-1],test_losses_dims_iwae[-1]])
        ax = vis2d.format_plot_axis(ax,ylabel='Model loss',xticks=(0,1,2,3,4),xticklabels=('Lattice-LVM (d=2)','VAE (d=2)','IWAE (d=2)','VAE (d=128)', 'IWAE (d=128)'))
        ax.tick_params('x',rotation=45)
        ##ax.boxplot(test_losses_dims_vae[0])
        #ax.boxplot(test_losses_dims_iwae[0])
        #ax.scatter(np.random.randn(*qmc_test_losses.shape)/8,qmc_test_losses)
        #ax.scatter(1+np.random.randn(*test_losses_dims_vae[0].shape)/8,test_losses_dims_vae[1])
        #ax.scatter(2+np.random.randn(*test_losses_dims_iwae[0].shape)/8,test_losses_dims_iwae[1])
        plt.savefig(os.path.join(stats_path,f'test_loss_violin_{dataset}.svg'))
        plt.show()
        plt.close()

if __name__ == '__main__':

    run_annoying_test_samples()