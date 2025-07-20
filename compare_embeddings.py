import torch
import umap
import os
from data.utils import load_data
from models.sampling import gen_fib_basis, gen_korobov_basis
from models.utils import *
from models.vae_base import VAE 
from models.qmc_base import QMCLVM
from train.losses import *
from train.model_saving_loading import *
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.optim import Adam
import matplotlib.pyplot as plt
import fire
import json

def compare_embeddings(model_save_loc,
                       dataset,
                       dataloc,
                       save_location,
                       batch_size=256,
                       lattice_m=15,
                       frames_per_sample=10
                       ):
    
    print("actually running code")
    save_location = os.path.join(save_location,dataset)
    if not os.path.isdir(save_location):
        os.mkdir(save_location)
    qmc_save_loc = os.path.join(model_save_loc,f'qmc_train_{dataset}_dim_comparison.tar')
    vae_2d_save_loc = os.path.join(model_save_loc,f'vae_train_{dataset}_dim_comparison_2d.tar')
    vae_128d_save_loc = os.path.join(model_save_loc,f'vae_train_{dataset}_dim_comparison_128d.tar')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qmc_latent_dim=3 if (('celeba' in dataset.lower()) or ('shapes3d' in dataset.lower())) else 2

    if qmc_latent_dim == 2:
        train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size,frames_per_sample=frames_per_sample)
        lattice = gen_fib_basis(m=lattice_m)
    else:
        train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size//2,frames_per_sample=frames_per_sample)

        lattice = gen_korobov_basis(a=1516,num_dims=qmc_latent_dim,num_points=4093)

    ##### Get latents from all models ########
    ### qmc ###
    qmc_lat_file = os.path.join(save_location,'qmc_latents.json')
    if not os.path.isfile(qmc_lat_file):
        print("Obtaining qmc latents...")
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

        print('done! saving...')
        qmc_latents = {'train':{'latents': train_latents_qmc.tolist(),'labels':train_labels_qmc.tolist()},
                    'test':{'latents': test_latents_qmc.tolist(),'labels':test_labels_qmc.tolist()}}
        qmc_model.to('cpu')
        with open(qmc_lat_file,'w') as f:
            json.dump(qmc_latents,f)
        
    else:
        print("loading qmc latents")
        with open(qmc_lat_file,'r') as f:
            qmc_latents = json.load(f)

        train_latents_qmc = np.array(qmc_latents['train']['latents'])
        train_labels_qmc = np.array(qmc_latents['train']['labels'])
        test_latents_qmc = np.array(qmc_latents['test']['latents'])
        test_labels_qmc = np.array(qmc_latents['test']['labels'])
        print("done!")
    #######################################################################################
    ### vae 2d ###
    vae2d_lat_file = os.path.join(save_location,'vae_2d_latents.json')
    if not os.path.isfile(vae2d_lat_file):
        print("Getting 2d VAE latents...")
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
        print("Done! saving out now")
        vae_latents_2d = {'train':{'latents': train_latents_vae_2d.tolist(),'labels':test_labels_vae_2d.tolist()},
                    'test':{'latents': test_latents_vae_2d.tolist(),'labels':train_labels_vae_2d.tolist()}}
        with open(vae2d_lat_file,'w') as f:
            json.dump(vae_latents_2d,f)
        print("done!")
    else:
        print("loading 2d VAE latents...")
        with open(vae2d_lat_file,'r') as f:
            vae_latents_2d = json.load(f)

        train_latents_vae_2d = np.array(vae_latents_2d['train']['latents'])
        train_labels_vae_2d = np.array(vae_latents_2d['train']['labels'])
        test_latents_vae_2d = np.array(vae_latents_2d['test']['latents'])
        test_labels_vae_2d = np.array(vae_latents_2d['test']['labels'])
    #################################################################################################
    ### vae 128d ###
    vae128d_lat_file = os.path.join(save_location,'vae_128d_latents.json')
    if not os.path.isfile(vae128d_lat_file):
        print("loading 128d latents...")
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
        vae_model_128d.to('cpu')

        print("done! saving out...")
        vae_latents_128d = {'train':{'latents': train_latents_vae_128d.tolist(),'labels':test_labels_vae_128d.tolist()},
                    'test':{'latents': test_latents_vae_128d.tolist(),'labels':train_labels_vae_128d.tolist()}}
        with open(vae128d_lat_file,'w') as f:
            json.dump(vae_latents_128d,f)
        print("done!!")
    else:
        print("loading 128d VAE latents...")
        with open(vae128d_lat_file,'r') as f:
            vae_latents_128d = json.load(f)

        train_latents_vae_128d = np.array(vae_latents_128d['train']['latents'])
        train_labels_vae_128d = np.array(vae_latents_128d['train']['labels'])
        test_latents_vae_128d = np.array(vae_latents_128d['test']['latents'])
        test_labels_vae_128d = np.array(vae_latents_128d['test']['labels'])
        print("done!")
    ######################################################################################################
    ###### umap vae 128d latents #############

    vae128d_umap_file = os.path.join(save_location,'vae_128d_umap.json')
    if not os.path.isfile(vae128d_umap_file):
        print("umappin' latents...")
        umap_model = umap.UMAP(n_components=2,random_state=128,n_neighbors=20,min_dist=0.1,n_jobs=len(os.sched_getaffinity(0)))
        umap_128_vae_train = umap_model.fit_transform(train_latents_vae_128d)
        umap_128_vae_test = umap_model.transform(test_latents_vae_128d)
        print("done")
        vae_128d_umap = {'train': umap_128_vae_train.tolist(),
                    'test':umap_128_vae_test.tolist()}
        with open(vae128d_umap_file,'w') as f:
            json.dump(vae_128d_umap,f)
    else:
        with open(vae128d_umap_file,'r') as f:

            vae_128d_umap = json.load(f)

        umap_128_vae_train = vae_128d_umap['train']
        umap_128_vae_test = vae_128d_umap['test']

    mosaic = [['QMC','QMC','VAE 2d', 'VAE 2d', 'VAE 128d', 'VAE 128d'],
              ['QMC','QMC','VAE 2d', 'VAE 2d', 'VAE 128d', 'VAE 128d']]
    
    fig,axs = plt.subplot_mosaic(mosaic,figsize=(22,7))

    axs['QMC'].scatter(train_latents_qmc[:,0],train_latents_qmc[:,1],c=train_labels_qmc,cmap='tab:10')
    axs['VAE 2d'].scatter(train_latents_vae_2d[:,0],train_latents_vae_2d[:,1],c=train_labels_vae_2d,cmap='tab:10')
    axs['VAE 128d'].scatter(umap_128_vae_train[:,0],umap_128_vae_train[:,1],c=train_labels_vae_128d,cmap='tab:10')

    plt.savefig(os.path.join(save_location,f'latent_rep_comparison_{dataset}.png'))
    plt.close()

if __name__ =='__main__':

    fire.Fire(compare_embeddings)