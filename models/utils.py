import torch
from models.layers import ResCellNVAESimple,ZeroLayer,PermutationLayer,PrintLayer,AVADecodeLayer,AVAEncodeLayer
import torch.nn as nn
from models.qmc_base import TorusBasis
from models.vae_base import Encoder

def get_decoder_arch(dataset_name,latent_dim,arch='qmc',n_per_sample=5):

    decoder = torch.nn.Sequential()
    if arch == 'qmc':
        latent_dim *= 2
    elif arch == 'conditional_qmc':
        latent_dim = latent_dim*2 + 1
    
    decoder.append(nn.Linear(latent_dim,2048))

    if 'mnist_simple' in dataset_name.lower():
        print("getting SHRIMPLE decoder")
        decoder = nn.Sequential(nn.Linear(latent_dim,500))
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
            ResCellNVAESimple(64,expand_factor=2,in_h=7,in_w=7),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), #nn.Linear(64*7*7,32*14*14),
            nn.Conv2d(64,32,1),
            ResCellNVAESimple(32,expand_factor=4,in_h=14,in_w=14),#nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1,groups=32),#nn.Linear(32*14*14,1*28*28),
            nn.Conv2d(32,16,1),
            ResCellNVAESimple(16,expand_factor=4,in_h=28,in_w=28),
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
            ResCellNVAESimple(64,expand_factor=2,in_h=5,in_w=5),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), #nn.Linear(64*5*5,64*10*10),
            nn.ConvTranspose2d(64,32,3,1,padding=1),
            ResCellNVAESimple(32,expand_factor=4,in_h=10,in_w=10),#nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1,groups=32),#nn.Linear(32*10*10,32*20*20),
            nn.ConvTranspose2d(32,16,3,1,padding=1),
            ResCellNVAESimple(16,expand_factor=8,in_h=20,in_w=20),#nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1,groups=16),#nn.Linear(16*20*20,16*40*40),
            nn.ConvTranspose2d(16,8,3,1,padding=1),
            ResCellNVAESimple(8,expand_factor=8,in_h=40,in_w=40),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1,groups=8),#nn.Linear(8*40*40,8*80*80),
            nn.ConvTranspose2d(8,4,3,1,padding=1),
            ResCellNVAESimple(4,expand_factor=8,in_h=80,in_w=80),
            ResCellNVAESimple(4,expand_factor=4,in_h=80,in_w=80),
            ResCellNVAESimple(4,expand_factor=2,in_h=80,in_w=80),
            nn.ConvTranspose2d(4,1,3,1,padding=1),
            nn.Sigmoid()]
 

    elif ('finch' in dataset_name.lower()):

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
        
    elif ('gerbil_ava' in dataset_name.lower()):
        decoder = nn.Sequential(nn.Linear(latent_dim,64))

        layers = [nn.ReLU(),
                  nn.Linear(64,256),
                  nn.ReLU(),
                  nn.Linear(256,1024),
                  nn.ReLU(),
                  nn.Linear(1024,8192),
                  nn.ReLU(),
                  nn.Unflatten(1,(32,16,16)),
                  AVADecodeLayer(32,24,16,16),
                  AVADecodeLayer(24,16,32,32),
                  AVADecodeLayer(16,8,64,64),
                  AVADecodeLayer(8,3,128,128), # doubles dimensionality (256x256)
                  nn.LayerNorm([3,256,256]),
                  nn.Conv2d(3,1,3,padding=1,stride=2),
                  #nn.ConvTranspose2d(3,1,3,1,padding=1),
                  #ResCellNVAESimple(1,expand_factor=32,in_h=128,in_w=128), # added these post processing layers to try to get things to work maybe a little bit better
                  #ResCellNVAESimple(1,expand_factor=32,in_h=128,in_w=128),
                  #ResCellNVAESimple(1,expand_factor=32,in_h=128,in_w=128),
                  nn.Sigmoid()]
        
    elif ('gerbil' in dataset_name.lower()):

        """
        older version used finch decoder
        """
        """
        middle version
        """
        layers = [nn.Linear(2048, 64*8*8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), # 64 x 8 x 8 -> 64 x 16 x 16
            nn.ConvTranspose2d(64,32,3,1,padding=1),
            ResCellNVAESimple(32,expand_factor=4,in_h=16,in_w=16),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1), # 32 x 16 x 16 -> 32 x 32 x 32
            nn.ConvTranspose2d(32,16,3,1,padding=1),
            ResCellNVAESimple(16,expand_factor=4,in_h=32,in_w=32),
            nn.ConvTranspose2d(16,16,3,stride=2,padding=1,output_padding=1), # 16 x 32 x 32 -> 16 x 64 x 64
            nn.ConvTranspose2d(16,8,3,1,padding=1),
            ResCellNVAESimple(8,expand_factor=4,in_h=64,in_w=64),
            nn.ConvTranspose2d(8,8,3,stride=2,padding=1,output_padding=1), # 8 x 64 x 64 -> 8 x 128 x 128
            nn.ConvTranspose2d(8,1,3,1,padding=1), # previously 8->1
            #nn.Sigmoid(),
            ] # maxpool added from previous versions
            #nn.MaxPool2d(3,stride=2,padding=1)
            #nn.Conv2d(8,4,1),
            #nn.ReLU(), # added from previous versions
            #nn.ConvTranspose2d(4,4,3,stride=2,padding=1,output_padding=1), # 4 x 128 x 128 -> 4 x 256 x 256 # added from previous versions
        """
        new version (from AVA)

        
        decoder = nn.Sequential(TorusBasis(),
                                nn.Linear(2*latent_dim,64)) if arch =='qmc' else nn.Sequential(nn.Linear(latent_dim,64))

        layers = [nn.ReLU(),
                  nn.Linear(64,256),
                  nn.ReLU(),
                  nn.Linear(256,1024),
                  nn.ReLU(),
                  nn.Linear(1024,8192),
                  nn.ReLU(),
                  nn.Unflatten(1,(32,16,16)),
                  nn.BatchNorm2d(32),
                  nn.ConvTranspose2d(32,24,3,1,padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(24),
                  nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(24),
                  nn.ConvTranspose2d(24,16,3,1,padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  nn.ConvTranspose2d(16,8,3,1,padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(8),
                  nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(8),
                  nn.ConvTranspose2d(8,1,3,1,padding=1)]
        """

    elif 'mocap_simple' in dataset_name.lower():
        decoder = nn.Sequential(nn.Linear(latent_dim,500))
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

        decoder = nn.Sequential(nn.Linear(latent_dim,500))
        layers = [
                nn.ReLU(),
                nn.Linear(500,1000),
                nn.Unflatten(1,(1,1,1000))
        ]

    elif 'shapes3d' in dataset_name.lower():
        """
        old version
        """
        """
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
        """
        """
        new version
        """
        layers = [nn.ReLU(),
                  nn.Linear(2048,2048),
                  nn.ReLU(),
                  nn.Linear(2048,512), # 512 = 32*4*4
                  nn.ReLU(),
                  nn.Unflatten(1,(32,4,4)),
                  AVADecodeLayer(32,24,4,4), # 24 x 8 x 8
                  AVADecodeLayer(24,16,8,8), # 16 x 16 x 16
                  AVADecodeLayer(16,8,16,16), # 8 x 32 x 32
                  AVADecodeLayer(8,3,32,32), # 3 x 64 x 64
                  ResCellNVAESimple(3,expand_factor=8,in_h=64,in_w=64),
                  ResCellNVAESimple(3,expand_factor=8,in_h=64,in_w=64),
                  ResCellNVAESimple(3,expand_factor=8,in_h=64,in_w=64),
                  nn.Sigmoid(),
                  PermutationLayer()]
        

    for layer in layers:
        decoder.append(layer)


    return decoder 


def get_encoder_arch(dataset_name,latent_dim,n_per_sample=5):


    if 'mnist_simple' in dataset_name.lower():
        print("getting SHRIMPLE encoder")
        encoder_net = nn.Flatten(start_dim=1,end_dim=-1)
        mu_net = nn.Sequential(nn.Linear(28**2,500),
                               nn.ReLU(),
                               nn.Linear(500,latent_dim))
        L_net = ZeroLayer()
        d_net = nn.Sequential(nn.Linear(28**2,500),
                               nn.ReLU(),
                               nn.Linear(500,latent_dim))
        enc = Encoder(net=encoder_net,mu_net=mu_net,l_net=L_net,d_net=d_net,latent_dim=latent_dim)
        #print(list(enc.named_parameters()))
    elif 'mnist' in dataset_name.lower():

        encoder_net =nn.Sequential(nn.Conv2d(1,16,1),
                           ResCellNVAESimple(16,expand_factor=1,in_h=28,in_w=28),
                            ResCellNVAESimple(16,expand_factor=2,in_h=28,in_w=28),
                            ResCellNVAESimple(16,expand_factor=4,in_h=28,in_w=28),
                           nn.Conv2d(16,32,1),#,stride=2,padding=1),
                           nn.Conv2d(32,32,3,stride=2,padding=1,groups=32),
                           ResCellNVAESimple(32,expand_factor=4,in_h=14,in_w=14),
                           nn.Conv2d(32,64,1),
                           nn.Conv2d(64,64,3,stride=2,padding=1,groups=64),
                           ResCellNVAESimple(64,expand_factor=2,in_h=7,in_w=7),
                           nn.Flatten(start_dim=1,end_dim=-1),
                           nn.Linear(64*7*7,2048),
                          nn.Tanh())
        mu_net = nn.Linear(2048,latent_dim)
        L_net = nn.Linear(2048,latent_dim)
        d_net = nn.Linear(2048,latent_dim) 

        enc = Encoder(net=encoder_net,mu_net=mu_net,l_net=L_net,d_net=d_net,latent_dim=latent_dim)


    elif 'celeba' in dataset_name.lower():

        #layers = [nn.ReLU(),
        #    nn.Linear(2048,64*5*5),
        #    nn.Unflatten(1, (64, 5, 5)),
        #    ResCellNVAESimple(64,expand_factor=2),
        #    nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), #nn.Linear(64*5*5,64*10*10),
        #    nn.Conv2d(64,32,1),
        #    ResCellNVAESimple(32,expand_factor=4),#nn.ReLU(),
        #    nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1,groups=32),#nn.Linear(32*10*10,32*20*20),
        #    nn.Conv2d(32,16,1),
        #    ResCellNVAESimple(16,expand_factor=8),#nn.ReLU(),
        #    nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1,groups=16),#nn.Linear(16*20*20,16*40*40),
        #    nn.Conv2d(16,8,1),
        #    ResCellNVAESimple(8,expand_factor=8),
        #    nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1,groups=8),#nn.Linear(8*40*40,8*80*80),
        #    nn.Conv2d(8,4,1),
        #    ResCellNVAESimple(4,expand_factor=8),
        #    ResCellNVAESimple(4,expand_factor=4),
        #    ResCellNVAESimple(4,expand_factor=2),
        #    nn.Conv2d(4,1,1),
        #    nn.Sigmoid()]
        encoder_net = nn.Sequential(nn.Conv2d(1,4,1),
                                    ResCellNVAESimple(4,expand_factor=2,in_h=80,in_w=80),
                                    ResCellNVAESimple(4,expand_factor=4,in_h=80,in_w=80),
                                    ResCellNVAESimple(4,expand_factor=8,in_h=80,in_w=80),
                                    nn.Conv2d(4,8,1),
                                    nn.Conv2d(8,8,3,stride=2,padding=1,groups=8),
                                    ResCellNVAESimple(8,expand_factor=8,in_h=40,in_w=40),
                                    nn.Tanh(),
                                    nn.Conv2d(8,16,1),
                                    nn.Conv2d(16,16,3,stride=2,padding=1,groups=16),
                                    ResCellNVAESimple(16,expand_factor=8,in_h=20,in_w=20),
                                    nn.Tanh(),
                                    nn.Conv2d(16,32,1),
                                    nn.Conv2d(32,32,3,stride=2,padding=1,groups=32),
                                    ResCellNVAESimple(32,expand_factor=4,in_h=10,in_w=10),
                                    nn.Tanh(),
                                    nn.Conv2d(32,64,1),
                                    nn.Conv2d(64,64,3,stride=2,padding=1,groups=64),
                                    ResCellNVAESimple(64,expand_factor=2,in_h=5,in_w=5),
                                    nn.Flatten(start_dim=1,end_dim=-1),
                                    nn.Linear(64*5*5,2048),
                                    nn.Tanh())
        """ old architecture
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
        """
        mu_net = nn.Linear(2048,latent_dim)
        L_net = nn.Linear(2048,latent_dim)
        d_net = nn.Linear(2048,latent_dim)
        

        enc = Encoder(encoder_net,mu_net,L_net,d_net,latent_dim)

    elif ('finch' in dataset_name.lower()):
        enc = Encoder(latent_dim=latent_dim)
    
    elif 'gerbil' in dataset_name.lower():
        """
        older versions
        """
        #enc = Encoder(latent_dim=latent_dim)
        """
        new version
        """
        encoder_net = nn.Sequential(nn.Conv2d(1,1,3,stride=2,padding=1),
                                    nn.Conv2d(1,8,1),
                                    ResCellNVAESimple(8,expand_factor=4,in_h=64,in_w=64),
                                    nn.Conv2d(8,8,3,stride=2,padding=1,groups=8),
                                    nn.Conv2d(8,16,1),
                                    ResCellNVAESimple(16,expand_factor=4,in_h=32,in_w=32),
                                    nn.Conv2d(16,16,3,stride=2,padding=1,groups=16),
                                    nn.Conv2d(16,32,1),
                                    ResCellNVAESimple(32,expand_factor=4,in_h=16,in_w=16),
                                    nn.Conv2d(32,32,3,stride=2,padding=1,groups=16),
                                    nn.Conv2d(32,64,1),
                                    nn.Flatten(start_dim=1,end_dim=-1),
                                    nn.Linear(64*8*8,2048),
                                    nn.Tanh())
        mu_net = nn.Linear(2048,latent_dim)
        L_net = nn.Linear(2048,latent_dim)
        d_net = nn.Linear(2048,latent_dim)
        

        enc = Encoder(encoder_net,mu_net,L_net,d_net,latent_dim)
    


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
            ResCellNVAESimple(3,expand_factor=8,in_h=64,in_w=64),
            ResCellNVAESimple(3,expand_factor=4,in_h=64,in_w=64),
            ResCellNVAESimple(3,expand_factor=2,in_h=64,in_w=64),
            nn.Conv2d(3,8,1),
            nn.Conv2d(8,8,3,stride=2,padding=1,groups=8),
            ResCellNVAESimple(8,expand_factor=4,in_h=32,in_w=32),
            nn.Conv2d(8,16,1),
            nn.Conv2d(16,16,3,stride=2,padding=1,groups=16),
            ResCellNVAESimple(16,expand_factor=2,in_h=16,in_w=16),
            nn.Conv2d(16,32,1),
            nn.Conv2d(32,32,3,stride=2,padding=1,groups=32),
            nn.Tanh(),
            nn.Conv2d(32,64,1),
            nn.Conv2d(64,64,4,stride=2,padding=1,groups=64),
            ResCellNVAESimple(64,expand_factor=1,in_h=4,in_w=4),
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(64*4*4,2048)
        )
        
        mu_net = nn.Linear(2048,latent_dim)
        L_net = nn.Linear(2048,latent_dim)
        d_net = nn.Linear(2048,latent_dim)

        enc = Encoder(encoder_net,mu_net,L_net,d_net,latent_dim)

    #print(list(enc.named_parameters()))
    return enc