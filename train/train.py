import numpy as np
from tqdm import tqdm
from torch.optim import Adam


def train_epoch(model,optimizer,loader,base_sequence,loss_function):
    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence)
        loss = loss_function(samples, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def train_loop(model,loader,base_sequence,loss_function,nEpochs=100):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in range(nEpochs):

        batch_loss,model,optimizer = train_epoch(model,optimizer,loader,base_sequence,loss_function)

        losses += batch_loss

        print(f'Epoch {epoch + 1} Average loss: {np.nanmean(batch_loss):.4f}')


    return model, optimizer,losses
        


    
