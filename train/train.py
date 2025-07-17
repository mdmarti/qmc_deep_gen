import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch
from model.losses import jacEnergy


def train_epoch(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in tqdm(enumerate(loader)):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def test_epoch(model,loader,base_sequence,loss_function):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    epoch_losses = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(model.device)
            samples = model(base_sequence)
            loss = loss_function(samples, data)
            test_loss += loss.item()
            epoch_losses.append(loss.item())

    return epoch_losses

def train_loop(model,loader,base_sequence,loss_function,nEpochs=100,verbose=False,
               random=True,mod=True):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):

        if verbose:
            batch_loss,model,optimizer = train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod)    
        else:
            batch_loss,model,optimizer = train_epoch(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod)

        losses += batch_loss
        if verbose:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')

    return model, optimizer,losses

def train_epoch_mc(model,optimizer,loader,mc_func,loss_function):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(loader):
        base_sequence=mc_func().to(model.device)
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random=False,mod=False)
        loss = loss_function(samples, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def train_loop_mc(model,loader,loss_function,mc_func,nEpochs=100,print_losses=False):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):

        batch_loss,model,optimizer = train_epoch_mc(model,optimizer,loader,mc_func,loss_function)

        losses += batch_loss
        if print_losses:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')


    return model, optimizer,losses


############ TO DO: FILL IN HERE, ADD REGULARIZER STUFF ####

def train_epoch_reg(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        reg = jacEnergy(samples)
        l = loss + reg
        l.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(l.item())

    return epoch_losses,model,optimizer

def train_epoch_reg_verbose(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in tqdm(enumerate(loader)):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        reg = jacEnergy(samples)
        l = loss + reg
        l.backward()
        train_loss += l.item()
        optimizer.step()
        epoch_losses.append(l.item())

    return epoch_losses,model,optimizer

def test_epoch_reg(model,loader,base_sequence,loss_function):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    epoch_losses = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(model.device)
            samples = model(base_sequence)
            loss = loss_function(samples, data)
            test_loss += loss.item()
            epoch_losses.append(loss.item())

    return epoch_losses

def train_loop_reg(model,loader,base_sequence,loss_function,nEpochs=100,verbose=False,
               random=True,mod=True):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):

        if verbose:
            batch_loss,model,optimizer = train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod)    
        else:
            batch_loss,model,optimizer = train_epoch(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod)

        losses += batch_loss
        if verbose:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')

    return model, optimizer,losses


        


    
