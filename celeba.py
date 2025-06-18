
import torch
from data.toy_dsets import *

from models.sampling import *
from models.qmc_base import *
from models.layers import *
from train.losses import gaussian_lp,gaussian_evidence,gaussian_elbo
from models.vae_base import *
import train.train_vae as train_vae
import train.train as train_qmc
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
from torch.optim import Adam
from train.model_saving_loading import *
from plotting.visualize import *
from data.bird_data import *

import fire



def run_celeba_experiments(save_location,dataloc,train_grid_m=16,test_grid_m=20,n_recons=50,nEpochs=300):



    ############ Set up experiment parameters ##################
    print('loading data...')
    train_ims = torch.load(os.path.join(dataloc,'train_80x80.pt'))
    test_ims = torch.load(os.path.join(dataloc,'test80x80.pt'))  
    print('done!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_workers = len(os.sched_getaffinity(0))
    print(f"using {n_workers} workers") 
    train_data = CelebADsetIms(train_ims)
    test_data = CelebADsetIms(test_ims)
    train_loader = DataLoader(train_data,batch_size=64,num_workers=n_workers,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=1,num_workers=n_workers,shuffle=False)
    #####################################################

    ############## QMC training #########################
    qmc_latent_dim=2
    
    decoder_qmc = nn.Sequential(
            TorusBasis(),
            nn.Linear(2*qmc_latent_dim,2048),
            #nn.ReLU(),
            #nn.Linear(2048, 32*7*7),
            nn.ReLU(),
            nn.Linear(2048,64*5*5),
            nn.Unflatten(1, (64, 5, 5)),
            ResCellNVAESimple(64,expand_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), #nn.Linear(64*5*5,64*10*10),
            nn.Conv2d(64,32,1),
            ResCellNVAESimple(32,expand_factor=4),#nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1,groups=32),#nn.Linear(32*10*10,32*20*20),
            nn.Conv2d(32,16,1),
            ResCellNVAESimple(16,expand_factor=8),#nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1,groups=16),#nn.Linear(16*20*20,16*40*40),
            nn.Conv2d(16,8,1),
            ResCellNVAESimple(8,expand_factor=8),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1,groups=8),#nn.Linear(8*40*40,8*80*80),
            nn.Conv2d(8,4,1),
            ResCellNVAESimple(4,expand_factor=8),
            ResCellNVAESimple(4,expand_factor=4),
            ResCellNVAESimple(4,expand_factor=2),
            nn.Conv2d(4,1,1),
            nn.Sigmoid(),
        )
    qmc_model = QMCLVM(latent_dim=qmc_latent_dim, device=device,decoder=decoder_qmc)
    qmc_loss_function = lambda samples,data: gaussian_evidence(samples,data,var=.1)
    
    train_base_sequence = gen_fib_basis(m=train_grid_m)
    test_base_sequence = gen_fib_basis(m=test_grid_m)

    save_qmc = os.path.join(save_location,'qmc_train_zebra_finch_experiment.tar')
    if not os.path.isfile(save_qmc):
        print("now training qmc model")
        ## starting this run takes a little while for some reason...
        qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop(qmc_model,train_loader,train_base_sequence.to(device),qmc_loss_function,nEpochs=nEpochs,print_losses=True)
        save(qmc_model.to('cpu'),qmc_opt,qmc_losses,fn=save_qmc)
        qmc_model.to(device)
    else:
        qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
        qmc_model,qmc_opt,qmc_losses = load(qmc_model,qmc_opt,save_qmc)

    qmc_losses = np.array(qmc_losses)
    ax = plt.gca()
    ax.plot(-np.array(qmc_losses))
    ax =  format_plot_axis(ax,ylabel='log evidence',xlabel='update number',xticks=ax.get_xticks(),yticks=ax.get_yticks())
    plt.savefig(os.path.join(save_location,'qmc_train_stats.svg'))
    plt.close()

    model_grid_plot(qmc_model.to(device),n_samples_dim=20,show=False,fn=os.path.join(save_location,'qmc_grid.png'))


    ######################################################


    ############## VAE training ##########################




    ######################################################


    ############### analysis and comparison ##############




    ######################################################

if __name__ == '__main__':

    fire.Fire(run_celeba_experiments)
