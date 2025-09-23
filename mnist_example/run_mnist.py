from torch import nn
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from data import *
from losses import *
from qlvm import *
from vae import *
from train import *
import matplotlib.pyplot as plt
import os
import fire 


def run_mnist(dataloc,
              save_plots=False,
              max_workers=4,
              batch_size=128):
    

    train_loader,test_loader = load_mnist(dataloc,max_workers,batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    qlvm = get_qlvm(device)
    qmc_lattice = gen_fib_basis(m=15)

    vae = get_vae(latent_dim=2,device=device)
    iwae = get_iwae(latent_dim=2,device=device)


    qlvm_loss =binary_evidence
    vae_loss = binary_elbo
    iwae_loss = binary_iwae_elbo

    qlvm,qlvm_opt,qlvm_losses = train_loop(qlvm,train_loader,qmc_lattice.to(qlvm.device),qlvm_loss,nEpochs=20)
    qlvm_test_loss = test_epoch(qlvm,test_loader,qmc_lattice.to(qlvm.device),qlvm_loss)

    vae,vae_opt,vae_losses = train_loop_vae(vae,train_loader,vae_loss,nEpochs=20)
    vae_test_loss = test_epoch_vae(vae,test_loader,vae_loss)
    
    iwae,iwae_opt,iwae_losses = train_loop_vae(iwae,train_loader,iwae_loss,nEpochs=20)
    iwae_test_loss = test_epoch_vae(iwae,test_loader,iwae_loss)

    if save_plots:
        ax = plt.gca()
        ax.scatter(np.zeros(qlvm_test_loss.shape),qlvm_test_loss)
        ax.scatter(np.ones(vae_test_loss.shape[1]),vae_test_loss.sum(axis=0))
        ax.scatter(2*np.ones(iwae_test_loss.shape[1]),iwae_test_loss.sum(axis=0))
        ax.set_ylabel("Model test loss")
        ax.set_xticks([0,1,2],['QLVM','VAE','IWAE'])
        
        plt.savefig(os.path.join(dataloc,'mnist_losses.png'))
        plt.close()

if __name__ == '__main__':

    fire.Fire(run_mnist)