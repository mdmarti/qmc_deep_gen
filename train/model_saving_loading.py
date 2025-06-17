import torch

def save(model,optimizer,run_info,fn = ''):
    """
    save a model. This will save model parameters,
    optimizer parameters, and info about the run
    the run info should include training epoch and 
    loss trajectory (in the case of VAEs, both the 
    reconstruction loss and the kl term)
    """
    
    model_state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()
    model_state = {'model': model_state_dict,'optimizer':opt_state_dict,'run info':run_info}
    torch.save(model_state,fn)

def load(model,optimizer,fn = ''):

    checkpoint = torch.load(fn,weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    run_info = checkpoint['run info']

    return model,optimizer, run_info