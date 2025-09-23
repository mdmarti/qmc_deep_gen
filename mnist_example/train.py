import torch
from tqdm import tqdm
from torch.optim import Adam
import numpy as np

######### VAE training #################

def train_epoch_vae(model,optimizer,loader,loss_function):

    model.train()
    train_loss = 0
    epoch_nlps,epoch_kls = [],[]
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(model.device)
        optimizer.zero_grad()
        recons,distribution = model(data)
        neg_lp,kl = loss_function(recons,distribution, data)
        loss = neg_lp + kl
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(),200)
        optimizer.step()
        epoch_nlps.append(neg_lp.item())
        epoch_kls.append(kl.item())

    return model,optimizer,epoch_nlps,epoch_kls

def test_epoch_vae(model,loader,loss_function):

    model.eval()
    test_loss = 0
    epoch_nlps,epoch_kls = [],[]
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(model.device).to(torch.float32)
            recons,distribution = model(data)
            neg_lp,kl = loss_function(recons,distribution, data)
            loss = neg_lp + kl
            test_loss += loss.item()
            epoch_nlps.append(neg_lp.item())
            epoch_kls.append(kl.item())

    return epoch_nlps,epoch_kls

def train_loop_vae(model,loader,loss_function,nEpochs=100):

    optimizer = Adam(model.parameters(),lr=1e-3)
    recons,kls = [],[]
    for epoch in tqdm(range(nEpochs)):

        model,optimizer,batch_recon,batch_kl = train_epoch_vae(model,optimizer,loader,loss_function)

        recons += batch_recon
        kls += batch_kl

    losses = [recons,kls]
    return model, optimizer,losses


######## QLVM Training ###########

def train_epoch(model,optimizer,loader,base_sequence,
                loss_function,random=True,mod=True):

    train_loss = 0
    epoch_losses = []
    for batch_idx,batch in enumerate(loader):
        data= batch[0]
        data = data.to(model.device)
        optimizer.zero_grad()

        samples = model(base_sequence,random,mod)

        loss = loss_function(samples,data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def test_epoch(model,loader,base_sequence,loss_function,
               random=True,mod=True,importance_weights=[]):

    test_loss = 0
    epoch_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            data = batch[0]
            data = data.to(model.device)

            samples = model(base_sequence,random=random,mod=mod)

            loss = loss_function(samples, data)

            test_loss += loss.item()
            epoch_losses.append(loss.item())

    return epoch_losses


def train_loop(model,loader,base_sequence,loss_function,nEpochs=100,
               random=True,mod=True):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):


        batch_loss,model,optimizer = train_epoch(model,optimizer,loader,base_sequence,loss_function,
                                                random=random,mod=mod)

        losses += batch_loss

        print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')

    return model, optimizer,losses