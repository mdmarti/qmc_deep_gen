import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch
from models.sampling import gen_samples_batch,get_default_voronoi


def train_epoch_adaptive(model,optimizer,loader,lattice,loss_function,log_prob,centered_voronoi_cell,conditional=False,n_samples_batch=-1):
    """
    Docstring for train_epoch_adaptive
    
    :param model: Description
    :param optimizer: Description
    :param loader: Description
    :param lattice: Description
    :param loss_function: Description
    :param log_prob: Description
    :param centered_voronoi_cell: Description
    :param conditional: Description
    :param n_samples_batch: Description
    """

    train_loss = 0
    epoch_losses = []
    if n_samples_batch < 1:
        n_samples_batch = len(lattice)
    #B = len(lattice)
    
    for batch_idx,batch in tqdm(enumerate(loader),total=len(loader)):
        data= batch[0]
        data = data.to(model.device)
        adaptive_samples,importance_weights = gen_samples_batch(lattice % 1, model, log_prob,data,n_samples_batch,centered_voronoi_cell)
        optimizer.zero_grad()
        if conditional:
            c = batch[1].to(torch.float32).to(model.device).view(1,-1)
            samples = model(adaptive_samples,random=False,mod=False,c=c)
        else:
            samples = model(adaptive_samples,random=False,mod=False)
        if len(importance_weights) == 0:
            loss = loss_function(samples, data)
        else:
            loss = loss_function(samples,data,importance_weights=importance_weights)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer


def val_epoch_adaptive(model,loader,lattice,loss_function,log_prob,conditional=False):
    """
    Docstring for val_epoch_adaptive
    
    :param model: Description
    :param loader: Description
    :param lattice: Description
    :param loss_function: Description
    :param log_prob: Description
    :param centered_voronoi_cell: Description
    :param conditional: Description
    """

    val_loss = 0
    epoch_losses = []
    B = len(lattice)
    model.eval()
    centered_cell = get_default_voronoi(lattice % 1).to(model.device)
    with torch.no_grad():
        for batch_idx,batch in tqdm(enumerate(loader),total=len(loader)):
            data= batch[0]
            data = data.to(model.device)
            adaptive_samples,importance_weights = gen_samples_batch(lattice % 1, model, log_prob,data,B,centered_cell)
            
            if conditional:
                c = batch[1].to(torch.float32).to(model.device).view(1,-1)
                samples = model(adaptive_samples,random=False,mod=False,c=c)
            else:
                samples = model(adaptive_samples,random=False,mod=False)
            if len(importance_weights) == 0:
                loss = loss_function(samples, data)
            else:
                loss = loss_function(samples,data,importance_weights=importance_weights)
            
            val_loss += loss.item()
            
            epoch_losses.append(loss.item())

    return epoch_losses

def train_loop_adaptive(model,loader,lattice,loss_function,log_prob,nEpochs=100,n_samples_batch=-1,
              conditional=False,verbose=False):
    """
    Docstring for train_loop_adaptive
    
    :param model: Description
    :param loader: Description
    :param lattice: Description
    :param loss_function: Description
    :param log_prob: Description
    :param nEpochs: Description
    :param n_samples_batch: Description
    :param conditional: Description
    :param verbose: Description
    """

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []

    centered_cell = get_default_voronoi(lattice % 1).to(model.device)
    for epoch in range(nEpochs):


        batch_loss,model,optimizer = train_epoch_adaptive(model,optimizer,loader,lattice,loss_function,log_prob,
                                                          centered_voronoi_cell=centered_cell,
                                             conditional=conditional,
                                                         n_samples_batch=n_samples_batch)

        losses += batch_loss
        if verbose:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')

    return model, optimizer,losses


    
