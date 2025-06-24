import torch
from models.layers import ResCellNVAESimple
import torch.nn as nn
from models.qmc_base import TorusBasis

def get_decoder_arch(dataset_name,latent_dim,arch='qmc'):

    decoder = torch.nn.Sequential()
    if arch == 'qmc':
        decoder.append(TorusBasis())
        decoder.append(nn.Linear(2*latent_dim,2048))
    else:
        decoder.append(nn.Linear(latent_dim,2048))

    if dataset_name.lower() == 'mnist':

        layers = []

    elif dataset_name.lower() == 'celeba':

        layers = [nn.ReLU(),
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
            nn.Sigmoid()]
 

    elif dataset_name.lower() == 'finch':

        layers = [nn.Linear(2048, 64*8*8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #nn.Linear(64*7*7,32*14*14),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),#nn.Linear(32*14*14,1*28*28),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()]
        

    for layer in layers:
        decoder.append(layer)


    return decoder 


def get_encoder_arch(dataset_name,latent_dim):


    if dataset_name.lower() == 'mnist':

        pass 

    elif dataset_name.lower() == 'celeba':

        pass 

    elif dataset_name.lower() == 'finch':
        pass
    return 