import torch
import os
from data.utils import load_data
from models.sampling import gen_fib_basis, gen_korobov_basis
from models.utils import *
import train_adaptive.train as train_asqlvm
from models.qmc_base import QMCLVM,TorusBasis
from train.losses import *
from train.model_saving_loading import *
from torch.optim import Adam
import plotting.visualize as vis2d # recon_comparison_plot,posterior_comparison_plot,qmc_train_plot,vae_train_plot,model_grid_plot
import fire
import json
from tqdm import tqdm

def run_qmc_experiments(save_location,dataloc,dataset,batch_size=1,
                            nEpochs=1,train_lattice_m=15,test_lattice_m=17,
                            frames_per_sample=1,
                            var=0.1,families=[2],latent_dim=2,n_models=5,
                            adaptive_samples=10):



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

    if latent_dim == 2:
        
        train_lattice = gen_fib_basis(m=train_lattice_m)
        test_lattice = gen_fib_basis(m=test_lattice_m)
    else:

        train_lattice = gen_korobov_basis(a=76,num_dims=latent_dim,num_points=1021)
        test_lattice = gen_korobov_basis(a=1487,num_dims=latent_dim,num_points=2039)


    basis = TorusBasis()

    assert (latent_dim ==2) or (latent_dim == 3), print(f"If training qmc model, latent dim must be 2 or 3, got {latent_dim}")

    saveloc = os.path.join(save_location, 'qmc_train_' + str(dataset) + '_' +str(latent_dim) + '_dim_comparison_{run:n}.tar')
    print(f"saving to {saveloc}")
    stats_save_loc = os.path.join(save_location,f'qmc_{latent_dim}_{n_models}_test_stats.json')

    test_lattice_m=18 if ('finch' in dataset.lower()) or ('gerbil' in dataset.lower()) else 20

    test_losses = []
    if not os.path.isfile(stats_save_loc):
        
        loss_func = binary_evidence if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda samples,data: gaussian_evidence(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
        lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 

        for model_num in range(n_models):
            print("*"*25)
            print(f"Now evaluating asqlvm {model_num}")
            print('*'*25)

            tmp_save_path = saveloc.format(run=model_num)
            decoder = get_decoder_arch(dataset_name=dataset,latent_dim=latent_dim,n_per_sample=frames_per_sample,arch='qmc')
            model = QMCLVM(latent_dim=latent_dim,device=device,decoder=decoder,basis=basis)

            if not os.path.isfile(tmp_save_path):
            

                model,opt,train_loss = train_asqlvm.train_loop_adaptive(model,train_loader,train_lattice.to(device),loss_func,\
                                                                    nEpochs=nEpochs,verbose='celeba' in dataset.lower(),n_samples_batch=adaptive_samples)
                print("Done training!")
                model.eval()
                with torch.no_grad():
                    
                    test_loss = train_asqlvm.val_epoch_adaptive(model,test_loader,test_lattice.to(device),loss_func)
                run_info = {'train':train_loss,'test':test_loss}
                save(model.to('cpu'),opt,run_info,fn=tmp_save_path)
                model.to(device)
                test_losses.append(np.sum(test_loss)/len(test_loader))
                vis2d.qmc_train_plot(train_loss,test_loss,save_fn=os.path.join(stats_save_loc,f'asqmc_{latent_dim}d_{dataset}_{model_num}_train_curve.svg'))
            else:
                opt = Adam(model.parameters(),lr=1e-3)
                model,opt,run_info = load(model,opt,tmp_save_path)
                model.eval()
                train_loss,test_loss = run_info['train'],run_info['test']
                test_losses.append(np.sum(test_loss)/len(test_loader))
   
        save_data = {'test_losses': test_losses}
        with open(stats_save_loc,'w') as f:
            json.dump(save_data,f)
    else:
        print("already trained & evaled! passing")
    

if __name__ =='__main__':

    fire.Fire(run_qmc_experiments)