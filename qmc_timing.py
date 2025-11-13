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
import time

def train_loop_timed(model,loader,base_sequence,loss_function,nEpochs=100,verbose=False,
               random=True,mod=True,conditional=False,importance_weights=[]):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    epoch_times = []
    for epoch in tqdm(range(nEpochs)):
        start = time.time()
        if verbose:
            batch_loss,model,optimizer = train_qmc.train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod,conditional=conditional,importance_weights=importance_weights)    
        else:
            batch_loss,model,optimizer = train_qmc.train_epoch(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod,conditional=conditional,importance_weights=importance_weights)
        end = time.time()
        epoch_times.append(end-start)
        losses += batch_loss
        if verbose:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')

    return model, optimizer,losses,epoch_times

def run_qmc_timing(save_location,dataloc,dataset,batch_size=256,
                            nEpochs=300,
                            frames_per_sample=10,
                            var=0.1,families=[2,4,5]):
    

    
    train_ms = range(10,17)

    train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size,frames_per_sample=frames_per_sample,
                                             families=families)


    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    latent_dim=2

    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    save_location = os.path.join(save_location,dataset)
    if not os.path.isdir(save_location):
        os.mkdir(save_location)
    
    test_lattice_m=18 if ('finch' in dataset.lower()) or ('gerbil' in dataset.lower()) else 20

    stats_save_loc = os.path.join(save_location,f'qmc_test_timing_stats.json')
    loss_func = binary_evidence if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda samples,data: gaussian_evidence(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
    lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 

    test_lattice = gen_fib_basis(m=test_lattice_m)
    avg_times = []
    avg_losses = []


    for train_lattice_m in train_ms:

        model_save_loc = os.path.join(save_location,f'qmc_m{train_lattice_m}.tar')
        train_lattice = gen_fib_basis(m=train_lattice_m)
        decoder = get_decoder_arch(dataset_name=dataset,latent_dim=2,arch='qmc',n_per_sample=1)
        model = QMCLVM(latent_dim=2,device=device,decoder=decoder)

        if not os.path.isfile(model_save_loc):
            model,opt,train_loss,times =  train_loop_timed(model,train_loader,train_lattice.to(device),loss_func,\
                                                                    nEpochs=nEpochs)
            timing = np.nanmean(times)
            model.eval()
            with torch.no_grad():
                test_loss = train_qmc.test_epoch(model,test_loader,test_lattice.to(device),loss_func)
            vae_run_info = {'train':train_loss,'test':test_loss}
            save(model.to('cpu'),opt,vae_run_info,fn=model_save_loc)
            model.to(device)
            #vis2d.vae_train_plot(train_loss,test_loss,save_fn=os.path.join(save_path,f'iwae_{k}k_{ii}_train_curve.svg'))
        else:
            model,opt,run_info = load(model,opt,model_save_loc)
            model.eval()
            train_loss,test_loss,timing = run_info['train'],run_info['test']
        
        avg_times.append(timing)    #test_losses.append(np.sum(test_loss)/len(test_loader))
        avg_losses.append(np.sum(test_loss)/len(test_loader))

    results_dict_qmc= {'timing': avg_times,
                    'losses': avg_losses}
    
    with open(stats_save_loc,'w') as f:
        json.dump(results_dict_qmc,f)

if __name__ =='__main__':

    fire.Fire(run_qmc_timing)
                