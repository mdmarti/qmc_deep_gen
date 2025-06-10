import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch


def train_epoch(model,optimizer,loader,loss_function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        optimizer.zero_grad()
        recons,distribution = model(data)
        loss = loss_function(recons,distribution, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def test_epoch(model,loader,loss_function):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    epoch_losses = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(device).to(torch.float32)
            recons,distribution = model(data)
            loss = loss_function(recons,distribution, data)
            test_loss += loss.item()
            epoch_losses.append(loss.item())

    return epoch_losses

def train_loop(model,loader,loss_function,nEpochs=100):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in range(nEpochs):

        batch_loss,model,optimizer = train_epoch(model,optimizer,loader,loss_function)

        losses += batch_loss

        print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')


    return model, optimizer,losses


        


    
