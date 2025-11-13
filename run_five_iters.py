import torch
import os
from data.utils import load_data
from models.sampling import gen_fib_basis, gen_korobov_basis
from models.utils import *
import train.train as train_qmc 
import train.train_vae as train_vae
from models.vae_base import VAE,IWAE
from models.qmc_base import QMCLVM,TorusBasis,GaussianICDFBasis
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
from tqdm import tqdm

def train_qmc_model(stats_save_loc,
                    model_save_loc,
                    train_lattice_m,
                    test_lattice_m,nEpochs,
                    train_loader,test_loader,
                    loss_fn,lp,
                    n_iters,
                    dataset,
                    latent_dim,
                    n_per_sample,model_type='qmc'):
    

    test_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if latent_dim == 2:
        
        train_lattice = gen_fib_basis(m=train_lattice_m)
        test_lattice = gen_fib_basis(m=test_lattice_m)
    else:

        train_lattice = gen_korobov_basis(a=76,num_dims=latent_dim,num_points=1021)
        test_lattice = gen_korobov_basis(a=1487,num_dims=latent_dim,num_points=2039)

    if model_type =='qmc':
        basis=TorusBasis()
    elif model_type=='gaussian_qmc':
        basis = GaussianICDFBasis()

    for ii in range(n_iters):
        print("*"*25)
        print(f"Now evaluating qmc {ii}")
        print('*'*25)
        tmp_save_path = model_save_loc.format(run=ii)
        decoder = get_decoder_arch(dataset_name=dataset,latent_dim=latent_dim,n_per_sample=n_per_sample,arch=model_type)
        model = QMCLVM(latent_dim=latent_dim,device=device,decoder=decoder,basis=basis)
        if not os.path.isfile(tmp_save_path):
            

            model,opt,train_loss = train_qmc.train_loop(model,train_loader,train_lattice.to(device),loss_fn,\
                                                                nEpochs=nEpochs,verbose='celeba' in dataset.lower())
            print("Done training!")
            model.eval()
            with torch.no_grad():
                
                test_loss = train_qmc.test_epoch(model,test_loader,test_lattice.to(device),loss_fn)
            run_info = {'train':train_loss,'test':test_loss}
            save(model.to('cpu'),opt,run_info,fn=tmp_save_path)
            model.to(device)
            test_losses.append(np.sum(test_loss)/len(test_loader))
            vis2d.qmc_train_plot(train_loss,test_loss,save_fn=os.path.join(stats_save_loc,f'qmc_{latent_dim}d_{dataset}_{ii}_train_curve.svg'))
        else:
            opt = Adam(model.parameters(),lr=1e-3)
            model,opt,run_info = load(model,opt,tmp_save_path)
            model.eval()
            train_loss,test_loss = run_info['train'],run_info['test']
            test_losses.append(np.sum(test_loss)/len(test_loader))
            #model.to(device)

    return test_losses

def train_vae_model(stats_save_loc,
                    model_save_loc,
                    nEpochs,
                    train_loader,test_loader,
                    loss_fn,lp,
                    n_iters,
                    dataset,
                    latent_dim,
                    n_per_sample,
                    diag):
    

    test_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ii in range(n_iters):
        print("*"*25)
        print(f"Now evaluating vae {ii}")
        print('*'*25)
        tmp_save_path = model_save_loc.format(run=ii)
        decoder = get_decoder_arch(dataset_name=dataset,latent_dim=latent_dim,arch='vae',n_per_sample=n_per_sample)
        encoder = get_encoder_arch(dataset_name=dataset,latent_dim=latent_dim,n_per_sample=n_per_sample,diag=diag)

        model = VAE(decoder=decoder,encoder=encoder,
                        distribution=LowRankMultivariateNormal,device=device)
        if not os.path.isfile(tmp_save_path):
            
            
            model,opt,train_loss = train_vae.train_loop(model,train_loader,loss_fn,nEpochs=nEpochs)
            model.eval()
            with torch.no_grad():
                test_loss = train_vae.test_epoch(model,test_loader,loss_fn)
            vae_run_info = {'train':train_loss,'test':test_loss}
            save(model.to('cpu'),opt,vae_run_info,fn=tmp_save_path)
            model.to(device)
            test_losses.append(np.sum(test_loss)/len(test_loader))
            vis2d.vae_train_plot(train_loss,test_loss,save_fn=os.path.join(stats_save_loc,f'vae_{latent_dim}d_{dataset}_{ii}_train_curve.svg'))
            
        else:
            opt = Adam(model.parameters(),lr=1e-3)
            model,opt,run_info = load(model,opt,tmp_save_path)
            model.eval()
            train_loss,test_loss = run_info['train'],run_info['test']
            test_losses.append(np.sum(test_loss)/len(test_loader))
            #model.to(device)

    return test_losses

def train_iwae_model(stats_save_loc,
                    model_save_loc,
                    nEpochs,
                    train_loader,test_loader,
                    loss_fn,lp,
                    n_iters,
                    dataset,
                    latent_dim,
                    n_per_sample,
                    diag,
                    k_samples=10):
    

    test_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ii in range(n_iters):
        print("*"*25)
        print(f"Now evaluating iwae {ii}")
        print('*'*25)
        tmp_save_path = model_save_loc.format(run=ii)
        decoder = get_decoder_arch(dataset_name=dataset,latent_dim=latent_dim,arch='vae',n_per_sample=n_per_sample)
        encoder = get_encoder_arch(dataset_name=dataset,latent_dim=latent_dim,n_per_sample=n_per_sample,diag=diag)

        model = IWAE(decoder=decoder,encoder=encoder,
                            distribution=LowRankMultivariateNormal,device=device,k_samples=k_samples)

        if not os.path.isfile(tmp_save_path):
            
            
            model,opt,train_loss = train_vae.train_loop(model,train_loader,loss_fn,nEpochs=nEpochs)
            model.eval()
            with torch.no_grad():
                test_loss = train_vae.test_epoch(model,test_loader,loss_fn)
            vae_run_info = {'train':train_loss,'test':test_loss}
            save(model.to('cpu'),opt,vae_run_info,fn=tmp_save_path)
            model.to(device)
            test_losses.append(np.sum(test_loss)/len(test_loader))
            vis2d.vae_train_plot(train_loss,test_loss,save_fn=os.path.join(stats_save_loc,f'vae_{latent_dim}d_{dataset}_{ii}_train_curve.svg'))
            
        else:
            opt = Adam(model.parameters(),lr=1e-3)
            model,opt,run_info = load(model,opt,tmp_save_path)
            model.eval()
            train_loss,test_loss = run_info['train'],run_info['test']
            test_losses.append(np.sum(test_loss)/len(test_loader))
            #model.to(device)

    return test_losses



def run_qmc_vae_experiments(save_location,dataloc,dataset,batch_size=256,
                            nEpochs=300,train_lattice_m=15,test_lattice_m=17,
                            frames_per_sample=1,
                            var=0.1,families=[2],model='qmc',latent_dim=3,n_iters=5,
                            k_samples=10,diag=False):



    ################ Shared Setup ######################################

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

    if 'qmc' in model:
        assert (latent_dim ==2) or (latent_dim == 3), print(f"If training qmc model, latent dim must be 2 or 3, got {latent_dim}")

        saveloc = os.path.join(save_location,'qmc_train_' + str(dataset) + '_' +str(latent_dim) + '_dim_comparison_{run:n}.tar')
        stats_save_loc = os.path.join(save_location,f'qmc_{latent_dim}_{n_iters}_test_stats.json')

        test_lattice_m=18 if ('finch' in dataset.lower()) or ('gerbil' in dataset.lower()) else 20
        if not os.path.isfile(stats_save_loc):
            
            loss_func = binary_evidence if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda samples,data: gaussian_evidence(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
            lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 

            test_losses = train_qmc_model(stats_save_loc=save_location,
                                          model_save_loc=saveloc,
                                          train_lattice_m=train_lattice_m,test_lattice_m=test_lattice_m,
                                          nEpochs=nEpochs,
                                          train_loader=train_loader,
                                          test_loader=test_loader,
                                          loss_fn=loss_func,
                                          lp=lp,
                                          n_iters=n_iters,
                                          dataset=dataset,latent_dim=latent_dim,n_per_sample=frames_per_sample,model_type=model)
            
            save_data = {'test_losses': test_losses}
            with open(stats_save_loc,'w') as f:
                json.dump(save_data,f)
        else:
            print("already trained & evaled! passing")
    
    elif model =='vae':
        saveloc = os.path.join(save_location,'vae_train_' + str(dataset) + '_' +str(latent_dim) + '_dim_comparison_{run:n}.tar')
        stats_save_loc = os.path.join(save_location,f'vae_{latent_dim}_{n_iters}_test_stats.json')

        loss_func = binary_elbo if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda recons,distribution,data: gaussian_elbo(recons,distribution,data,recon_precision=1/var) #or ('gerbil' in dataset.lower()) 
        lp = binary_lp if ('mnist' in dataset.lower())  or ('gerbil' in dataset.lower()) else lambda target,recon: gaussian_lp(recon,target,var=var) #or ('gerbil' in dataset.lower())
        test_losses = train_vae_model(stats_save_loc=save_location,
                                          model_save_loc=saveloc,
                                          nEpochs=nEpochs,
                                          train_loader=train_loader,
                                          test_loader=test_loader,
                                          loss_fn=loss_func,
                                          lp=lp,
                                          n_iters=n_iters,
                                          dataset=dataset,latent_dim=latent_dim,n_per_sample=frames_per_sample,
                                          diag=diag)
            
        save_data = {'test_losses': test_losses}
        with open(stats_save_loc,'w') as f:
            json.dump(save_data,f)

    elif model == 'iwae':
        saveloc = os.path.join(save_location,'iwae_train_' + str(dataset) + '_' +str(latent_dim) + '_dim_comparison_{run:n}.tar')
        stats_save_loc = os.path.join(save_location,f'iwae_{latent_dim}_{n_iters}_test_stats.json')

        loss_func = binary_iwae_elbo if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda recons,distribution,data: gaussian_iwae_elbo(recons,distribution,data,recon_precision=1/var) #or ('gerbil' in dataset.lower()) 
        lp = binary_lp if ('mnist' in dataset.lower())  or ('gerbil' in dataset.lower()) else lambda target,recon: gaussian_lp(recon,target,var=var) #or ('gerbil' in dataset.lower())
        test_losses = train_iwae_model(stats_save_loc=save_location,
                                          model_save_loc=saveloc,
                                          nEpochs=nEpochs,
                                          train_loader=train_loader,
                                          test_loader=test_loader,
                                          loss_fn=loss_func,
                                          lp=lp,
                                          n_iters=n_iters,
                                          k_samples=k_samples,
                                          dataset=dataset,latent_dim=latent_dim,n_per_sample=frames_per_sample,
                                          diag=diag)
            
        save_data = {'test_losses': test_losses}
        with open(stats_save_loc,'w') as f:
            json.dump(save_data,f)

    else:
        raise NotImplementedError
    

if __name__ =='__main__':

    fire.Fire(run_qmc_vae_experiments)

        





