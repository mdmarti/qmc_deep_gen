import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch


def train_epoch(model,optimizer,loader,loss_function):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def test_epoch(model,loader,loss_function):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def train_loop(model,loader,loss_function,nEpochs=100):

    optimizer = Adam(model.parameters(),lr=1e-3)
    recons,kls = [],[]
    for epoch in tqdm(range(nEpochs)):

        model,optimizer,batch_recon,batch_kl = train_epoch(model,optimizer,loader,loss_function)

        recons += batch_recon
        kls += batch_kl

        #print(f'Epoch {epoch + 1} Average loss: {(np.sum(batch_recon) + np.sum(batch_kl))/len(loader.dataset):.4f}')

    losses = [recons,kls]
    return model, optimizer,losses


        


    
