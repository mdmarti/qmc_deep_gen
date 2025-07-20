import torch
from models.layers import ResCellNVAESimple,ZeroLayer,PermutationLayer
import torch.nn as nn
from models.qmc_base import TorusBasis
from models.vae_base import Encoder

def get_decoder_arch(dataset_name,latent_dim,arch='qmc',n_per_sample=5):

    decoder = torch.nn.Sequential()
    if arch == 'qmc':
        decoder.append(TorusBasis())
        decoder.append(nn.Linear(2*latent_dim,2048))
    else:
        decoder.append(nn.Linear(latent_dim,2048))

    if 'mnist_simple' in dataset_name.lower():

        decoder = nn.Sequential(TorusBasis(),
                                nn.Linear(2*latent_dim,500)) if arch =='qmc' else nn.Sequential(nn.Linear(latent_dim,500))
        layers = [
                nn.ReLU(),
                nn.Linear(500,28**2),
                nn.Sigmoid(),
                nn.Unflatten(1,(1,28,28))
        ]
    elif 'mnist' in dataset_name.lower():

        layers = [nn.ReLU(),
            nn.Linear(2048,64*7*7),
            nn.Unflatten(1, (64, 7, 7)),
            ResCellNVAESimple(64,expand_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), #nn.Linear(64*7*7,32*14*14),
            nn.Conv2d(64,32,1),
            ResCellNVAESimple(32,expand_factor=4),#nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1,groups=32),#nn.Linear(32*14*14,1*28*28),
            nn.Conv2d(32,16,1),
            ResCellNVAESimple(16,expand_factor=4),
            #ResCellNVAESimple(16,expand_factor=2),
            #ResCellNVAESimple(16,expand_factor=1),
            nn.Conv2d(16,1,1),
            nn.Sigmoid()]

    

    elif 'celeba' in dataset_name.lower():

        """
        new architecture
        layers = [nn.ReLU(),
            nn.Linear(2048,64*5*5),
            nn.Unflatten(1, (64, 5, 5)),
            ResCellNVAESimple(64,expand_factor=6),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), #nn.Linear(64*5*5,64*10*10),
            nn.Conv2d(64,32,1),
            ResCellNVAESimple(32,expand_factor=6),#nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1,groups=32),#nn.Linear(32*10*10,32*20*20),
            nn.Conv2d(32,16,1),
            ResCellNVAESimple(16,expand_factor=6),#nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1,groups=16),#nn.Linear(16*20*20,16*40*40),
            nn.Conv2d(16,8,1),
            ResCellNVAESimple(8,expand_factor=6),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1,groups=8),#nn.Linear(8*40*40,8*80*80),
            #nn.Conv2d(8,4,1),
            ResCellNVAESimple(8,expand_factor=6),
            ResCellNVAESimple(8,expand_factor=6),
            ResCellNVAESimple(8,expand_factor=6),
            nn.Conv2d(8,1,1),
            nn.Sigmoid()]
        """
        """
        old architecture
        """
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
 

    elif 'finch' in dataset_name.lower():

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
        
    elif 'mocap_simple' in dataset_name.lower():
        decoder = nn.Sequential(TorusBasis(),
                                nn.Linear(2*latent_dim,500)) if arch =='qmc' else nn.Sequential(nn.Linear(latent_dim,500))
        layers = [
                nn.ReLU(),
                nn.Linear(500,n_per_sample*100),
                nn.Unflatten(1,(1,n_per_sample,100))
        ]
        
    elif 'mocap' in dataset_name.lower():

         layers = [
                nn.ReLU(),
                nn.Linear(2048,2*n_per_sample*100),
                nn.ReLU(),
                nn.Linear(2*n_per_sample*100,n_per_sample*100),
                #nn.Sigmoid(),
                nn.Unflatten(1,(1,n_per_sample,100))
        ]
         
    elif 'blobs' in dataset_name.lower() or 'moons' in dataset_name.lower():

        decoder = nn.Sequential(TorusBasis(),
                                nn.Linear(2*latent_dim,500)) if arch =='qmc' else nn.Sequential(nn.Linear(latent_dim,500))
        layers = [
                nn.ReLU(),
                nn.Linear(500,1000),
                nn.Unflatten(1,(1,1,1000))
        ]

    elif 'shapes3d' in dataset_name.lower():

        layers = [nn.ReLU(),
            nn.Linear(2048,64*4*4),
            nn.Unflatten(1, (64, 4, 4)),
            ResCellNVAESimple(64,expand_factor=1),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), #nn.Linear(64*4*4,64*8*8),
            nn.Conv2d(64,32,1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1,groups=32),#nn.Linear(32*8*8,32*16*16),
            nn.Conv2d(32,16,1),
            ResCellNVAESimple(16,expand_factor=2),#nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1,groups=16),#nn.Linear(16*32*32,16*32*32),
            nn.Conv2d(16,8,1),
            ResCellNVAESimple(8,expand_factor=4),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1,groups=8),#nn.Linear(8*32*32,8*64*64),
            nn.Conv2d(8,3,1),
            ResCellNVAESimple(3,expand_factor=8),
            ResCellNVAESimple(3,expand_factor=4),
            ResCellNVAESimple(3,expand_factor=2),
            #nn.Conv2d(4,1,1),
            nn.Sigmoid(),
            PermutationLayer()]
        

    for layer in layers:
        decoder.append(layer)


    return decoder 


def get_encoder_arch(dataset_name,latent_dim,n_per_sample=5):


    if 'mnist_simple' in dataset_name.lower():
        encoder_net = nn.Flatten(start_dim=1,end_dim=-1)
        mu_net = nn.Sequential(nn.Linear(28**2,500),
                               nn.ReLU(),
                               nn.Linear(500,latent_dim))
        L_net = ZeroLayer()
        d_net = nn.Sequential(nn.Linear(28**2,500),
                               nn.ReLU(),
                               nn.Linear(500,latent_dim))
        enc = Encoder(net=encoder_net,mu_net=mu_net,l_net=L_net,d_net=d_net,latent_dim=latent_dim)
        
    if 'mnist' in dataset_name.lower():

        encoder_net =nn.Sequential(nn.Conv2d(1,16,1),
                           ResCellNVAESimple(16,expand_factor=1),
                            ResCellNVAESimple(16,expand_factor=2),
                            ResCellNVAESimple(16,expand_factor=4),
                           nn.Conv2d(16,32,1),#,stride=2,padding=1),
                           nn.Conv2d(32,32,3,stride=2,padding=1,groups=32),
                           ResCellNVAESimple(32,expand_factor=4),
                           nn.Conv2d(32,64,1),
                           nn.Conv2d(64,64,3,stride=2,padding=1,groups=64),
                           ResCellNVAESimple(64,expand_factor=2),
                           nn.Flatten(start_dim=1,end_dim=-1),
                           nn.Linear(64*7*7,2048),
                          nn.Tanh())
        mu_net = nn.Linear(2048,latent_dim)
        L_net = nn.Linear(2048,latent_dim)
        d_net = nn.Linear(2048,latent_dim) 

        enc = Encoder(net=encoder_net,mu_net=mu_net,l_net=L_net,d_net=d_net,latent_dim=latent_dim)


    elif 'celeba' in dataset_name.lower():

        encoder_net =nn.Sequential(nn.Conv2d(1,8,1),#,stride=2,padding=1),
                            ResCellNVAESimple(8,expand_factor=6),
                            ResCellNVAESimple(8,expand_factor=6),
                            ResCellNVAESimple(8,expand_factor=6),
                            #nn.Conv2d(8,8,1),
                            nn.Conv2d(8,8,3,stride=2,padding=1,groups=8),
                            ResCellNVAESimple(8,expand_factor=6),
                            nn.Tanh(),
                            nn.Conv2d(8,16,1),
                            nn.Conv2d(16,16,3,stride=2,padding=1,groups=16),
                            ResCellNVAESimple(16,expand_factor=6),
                            nn.Tanh(),
                            nn.Conv2d(16,32,1),
                            nn.Conv2d(32,32,3,stride=2,padding=1,groups=32),
                            ResCellNVAESimple(32,expand_factor=6),
                            nn.Tanh(),
                            nn.Conv2d(32,64,1),
                            nn.Conv2d(64,64,3,stride=2,padding=1,groups=64),
                            ResCellNVAESimple(64,expand_factor=6),
                            nn.Flatten(start_dim=1,end_dim=-1),
                            nn.Linear(64*5*5,2048),
                            nn.Tanh())
        mu_net = nn.Linear(2048,latent_dim)
        L_net = nn.Linear(2048,latent_dim)
        d_net = nn.Linear(2048,latent_dim)

        enc = Encoder(encoder_net,mu_net,L_net,d_net,latent_dim)

    elif 'finch' in dataset_name.lower():
        enc = Encoder(latent_dim=latent_dim)

    elif 'mocap_simple' in dataset_name.lower():
        encoder_net = nn.Flatten(start_dim=1,end_dim=-1)
        mu_net = nn.Sequential(nn.Linear(n_per_sample*100,500),
                               nn.ReLU(),
                               nn.Linear(500,latent_dim))
        L_net = ZeroLayer()
        d_net = nn.Sequential(nn.Linear(n_per_sample*100,500),
                               nn.ReLU(),
                               nn.Linear(500,latent_dim))
        enc = Encoder(net=encoder_net,mu_net=mu_net,l_net=L_net,d_net=d_net,latent_dim=latent_dim)

    elif 'mocap' in  dataset_name.lower():

        encoder_net = nn.Sequential(nn.Flatten(start_dim=1,end_dim=-1),
                                    nn.Linear(n_per_sample*100,2*n_per_sample*100),
                                    nn.ReLU(),
                                    nn.Linear(2*n_per_sample*100,2048),
                                    nn.ReLU()
                                    )
        mu_net = nn.Linear(2048,latent_dim)
        L_net = nn.Linear(2048,latent_dim)
        d_net = nn.Linear(2048,latent_dim)

        enc = Encoder(encoder_net,mu_net,L_net,d_net,latent_dim)
        
    elif 'blobs' in dataset_name.lower() or 'moons' in dataset_name.lower():

        encoder_net = nn.Sequential(nn.Flatten(start_dim=1,end_dim=-1),
                                    nn.Linear(1000,500)
                                    )
        mu_net = nn.Linear(500,latent_dim)
        L_net = nn.Linear(500,latent_dim)
        d_net = nn.Linear(500,latent_dim)

        enc = Encoder(encoder_net,mu_net,L_net,d_net,latent_dim)


    elif 'shapes3d' in dataset_name.lower():

        encoder_net = nn.Sequential(
            PermutationLayer(permute_type='input'),
            ResCellNVAESimple(3,expand_factor=8),
            ResCellNVAESimple(3,expand_factor=4),
            ResCellNVAESimple(3,expand_factor=2),
            nn.Conv2d(3,8,1),
            nn.Conv2d(8,8,3,stride=2,padding=1,groups=8),
            ResCellNVAESimple(8,expand_factor=4),
            nn.Conv2d(8,16,1),
            nn.Conv2d(16,16,3,stride=2,padding=1,groups=16),
            ResCellNVAESimple(16,expand_factor=2),
            nn.Conv2d(16,32,1),
            nn.Conv2d(32,32,3,stride=2,padding=1,groups=32),
            nn.Tanh(),
            nn.Conv2d(32,64,1),
            nn.Conv2d(64,64,4,stride=2,padding=1,groups=64),
            ResCellNVAESimple(64,expand_factor=1),
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(64*4*4,2048)
        )
        
        mu_net = nn.Linear(2048,latent_dim)
        L_net = nn.Linear(2048,latent_dim)
        d_net = nn.Linear(2048,latent_dim)

        enc = Encoder(encoder_net,mu_net,L_net,d_net,latent_dim)

    return enc