import torch
from torch.utils.data import Dataset,DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import os

##### Factors for shapes3d ######
_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                        'scale': 8, 'shape': 4, 'orientation': 15}
#########

class CelebADsetIms(Dataset):

    """
    assumes that you're using data loaded directly in all at once,
    rather than loading 1 image at a time
    """

    def __init__(self,ims):

        self.ims = ims
        self.len = ims.shape[0]
    
    def __len__(self):

        return self.len

    def __getitem__(self,index):

        return (self.ims[index],[])
    
class GeneralToyDset(Dataset):

    """
    assumes data come from an sklearn dataset,
    but really encompasses anything that's a data/label pair
    """

    def __init__(self,ims,labels):

        self.ims = ims
        self.labels = labels

        self.len = ims.shape[0]

    def __len__(self):

        return self.len
    
    def __getitem__(self,index):

        return (self.ims[index],self.labels[index])
    

def generate_moons(n_samples,seed,noise_sd_in,noise_sd_out,test_size=0.2):
    
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise_sd_in, random_state=seed)
    z,y = noisy_moons

    gen = np.random.default_rng(seed=seed)

    w = gen.standard_normal(size=(2,1000))
    x = z @ w + gen.standard_normal(size=(z.shape[0],))*noise_sd_out

    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=test_size)

    return GeneralToyDset(train_x,train_y), GeneralToyDset(test_x,test_y)



def generate_blobs(n_samples,dim,seed,noise_sd_in,noise_sd_out,test_size=0.2):

    noisy_blobs = datasets.make_blobs(n_samples=n_samples,n_features=dim,random_state=seed,cluster_sd = noise_sd_in)

    z,y = noisy_blobs

    gen = np.random.default_rng(seed=seed)

    w = gen.standard_normal(size=(dim,1000))
    x = z @ w + gen.standard_normal(size=(z.shape[0],))*noise_sd_out

    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=test_size)

    return GeneralToyDset(train_x,train_y), GeneralToyDset(test_x,test_y)

def get_3d_shapes(dpath,seed,test_size=0.2):

    dataset = h5py.File(os.path.join(dpath,'3dshapes.h5'),'r')
    images,labels = dataset['images'],dataset['labels']
    (B,H,W,C) = images.shape
    images /= 255
    images = np.swapaxes(images.astype(np.float32),axis1=1,axis2=3) # B C W H
    images = np.swapaxes(images.astype(np.float32),axis1=2,axis2=3) # B C H W
    
    gen = np.random.default_rng(seed=seed)

    order = gen.choice(B,B,replace=False)
    train_end = int(round(B * (1-test_size)))
    train_x,train_y = images[order[:train_end]],labels[order[:train_end]]

    test_x,test_y = images[order[train_end:]],labels[order[train_end:]]

    return GeneralToyDset(train_x,train_y), GeneralToyDset(test_x,test_y)