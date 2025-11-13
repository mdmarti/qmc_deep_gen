import torch
import os
from data.utils import load_data
from models.sampling import gen_fib_basis, gen_korobov_basis
from models.utils import *
#import train.train as train_qmc 
from train.train_vae import *
from models.vae_base import VAE,IWAE
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
from tqdm import tqdm


def sample_loguniform(low,high):

    return np.exp(np.random.uniform(np.log(low), np.log(high)))

def train_loop_with_opt(model,optimizer,loader,loss_function,nEpochs=100):

    recons,kls = [],[]
    for epoch in tqdm(range(nEpochs)):

        model,optimizer,batch_recon,batch_kl = train_epoch(model,optimizer,loader,loss_function)

        recons += batch_recon
        kls += batch_kl

        #print(f'Epoch {epoch + 1} Average loss: {(np.sum(batch_recon) + np.sum(batch_kl))/len(loader.dataset):.4f}')

    losses = [recons,kls]
    return model, optimizer,losses

def get_vae_arch(n_hidden,latent_dim,hidden_dim,device):

    if n_hidden == 0:
        ### decoder ###
        #decoder = nn.Sequential(nn.Linear(latent_dim,28**2),
        #                        nn.Sigmoid(),
        #                        nn.Unflatten(1,(1,28,28)))
        ### encoder ###
        encoder_net = nn.Flatten(start_dim=1,end_dim=-1)
        mu_net = nn.Linear(28**2,latent_dim)
        L_net = ZeroLayer(28**2,latent_dim)
        d_net = nn.Linear(28**2,latent_dim)
        encoder = Encoder(net=encoder_net,mu_net=mu_net,l_net=L_net,d_net=d_net,latent_dim=latent_dim)

        
    else:
        ### decoder ###
        #decoder = nn.Sequential(nn.Linear(latent_dim,hidden_dim))
        #hidden_layers = []
        
        #for _ in range(n_hidden - 1):
        #    decoder.append(nn.ReLU())
        #    decoder.append(nn.Linear(hidden_dim,hidden_dim))
        #decoder.append(nn.ReLU())
        #decoder.append(nn.Linear(hidden_dim,28**2))
        #decoder.append(nn.Sigmoid())
        #decoder.append(nn.Unflatten(1,(1,28,28)))

        ### encoder ###
        encoder_net = nn.Flatten(start_dim=1,end_dim=-1)
        mu_net = nn.Sequential(nn.Linear(28**2,hidden_dim))
        L_net = ZeroLayer(28**2,latent_dim)
        d_net = nn.Sequential(nn.Linear(28**2,hidden_dim))
        for _ in range(n_hidden - 1):
            mu_net.append(nn.ReLU())
            mu_net.append(nn.Linear(hidden_dim,hidden_dim))

            d_net.append(nn.ReLU())
            d_net.append(nn.Linear(hidden_dim,hidden_dim))
        mu_net.append(nn.ReLU())
        mu_net.append(nn.Linear(hidden_dim,latent_dim))
        d_net.append(nn.ReLU())
        d_net.append(nn.Linear(hidden_dim,latent_dim))
        encoder = Encoder(net=encoder_net,mu_net=mu_net,l_net=L_net,d_net=d_net,latent_dim=latent_dim)
        decoder = nn.Sequential(nn.Linear(latent_dim,500))
        layers = [
                nn.Tanh(),
                nn.Linear(500,28**2),
                nn.Sigmoid(),
                nn.Unflatten(1,(1,28,28))
        ]
        for l in layers:
            decoder.append(l)
    model = VAE(decoder=decoder,encoder=encoder,
                    distribution=LowRankMultivariateNormal,device=device)
    
    return model


def vae_2d_hyperparam_cv(dataloc,save_path,n_models=50):


    hidden_layer_opts = [1,2,3,4,5]
    min_lr,max_lr=1e-5,1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim=2
    hidden_dim=500
    loss_func = binary_elbo
    lp = binary_lp 
    nEpochs=30

    train_loader,test_loader = load_data('mnist',dataloc,batch_size=128,frames_per_sample=1,
                                             families=[1])

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    model_dict= {f'run {ii}':{} for ii in range(n_models)}

    best_ind = -1
    best_loss = 1e5

    for ii in range(n_models):

        ind = f'run {ii}'

        n_hidden_layers = np.random.choice(hidden_layer_opts)
        encoder_lr = sample_loguniform(low=min_lr,high=max_lr)
        decoder_lr = sample_loguniform(low=min_lr,high=max_lr)

        print(f"Training model {ii+1} with {n_hidden_layers} hidden layers, encoder lr = {encoder_lr:.5f}, decoder lr = {decoder_lr:.5f}")

        model_save_loc=os.path.join(save_path,f'mnist_cv_run_{ii}.tar')

        model = get_vae_arch(n_hidden=n_hidden_layers,latent_dim=latent_dim,hidden_dim=hidden_dim,device=device)
        opt = Adam([{'params': model.encoder.parameters(),'lr':encoder_lr},
                        {'params':model.decoder.parameters(),'lr':decoder_lr}])
        
        if not os.path.isfile(model_save_loc):
            model,opt,train_loss =  train_loop_with_opt(model,opt,train_loader,loss_func,nEpochs=nEpochs)
            model.eval()
            with torch.no_grad():
                test_loss = test_epoch(model,test_loader,loss_func)
            vae_run_info = {'train':train_loss,'test':test_loss}
            save(model.to('cpu'),opt,vae_run_info,fn=model_save_loc)
            model.to(device)
            vis2d.vae_train_plot(train_loss,test_loss,save_fn=os.path.join(save_path,f'vae_cv_run_{ii}_train_curve.svg'))

        else:
            model,opt,run_info = load(model,opt,model_save_loc)
            model.eval()
            train_loss,test_loss = run_info['train'],run_info['test']
            #test_losses.append(np.sum(test_loss)/len(test_loader))
        avg_test_loss = np.sum(test_loss)/len(test_loader)
        if avg_test_loss < best_loss:
            best_ind = ii
        model_dict[ind]['test loss'] = avg_test_loss
        model_dict[ind]['n hidden'] = n_hidden_layers
        model_dict[ind]['encoder lr'] = encoder_lr
        model_dict[ind]['decoder lr'] = decoder_lr

        print(f"Avg test loss for model {ii+1} with {n_hidden_layers} hidden layers, encoder lr = {encoder_lr:.5f}, decoder lr = {decoder_lr:.5f}: {np.sum(test_loss)/len(test_loader):.4f}")
    print(f"Best model: {best_ind}")
    stats = model_dict[f"run {best_ind}"]
    print(f"Num hidden layers = {stats['n hidden']}")
    print(f"Encoder lr = {stats['encoder lr']:.5f}")
    print(f"Decoder lr = {stats['decoder lr']:.5f}")
    print(f"test loss: {stats['test loss']}")
    stats_save_loc = os.path.join(save_path,'vae_hyperparam_testlosses.json')
    with open(stats_save_loc,'w') as f:
        json.dump(model_dict,f)

if __name__ == '__main__':

    fire.Fire(vae_2d_hyperparam_cv)


