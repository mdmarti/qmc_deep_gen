import torch
from torch.utils.data import Dataset,DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import os
from torchvision import transforms

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

    def __init__(self,ims,labels,indices=[],transform = torch.from_numpy):

        self.ims = ims
        self.labels = labels
        if len(indices) == 0:
            indices = np.arange(len(ims))
        self.indices = indices

        self.len = indices.shape[0]
        self.transform = transform

    def __len__(self):

        return self.len
    
    def __getitem__(self,index):

        index = self.indices[index]

        return (self.transform(self.ims[index]),self.labels[index])
    

def generate_moons(n_samples,seed,noise_sd_in,noise_sd_out,test_size=0.2):
    
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise_sd_in, random_state=seed)
    z,y = noisy_moons

    gen = np.random.default_rng(seed=seed)

    w = gen.standard_normal(size=(2,1000))
    x = z @ w + gen.standard_normal(size=(z.shape[0],))*noise_sd_out

    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=test_size)

    return GeneralToyDset(train_x,train_y), GeneralToyDset(test_x,test_y)



def generate_blobs(n_samples,dim,seed,noise_sd_in,noise_sd_out,test_size=0.2):

    noisy_blobs = datasets.make_blobs(n_samples=n_samples,n_features=dim,random_state=seed,cluster_std = noise_sd_in)

    z,y = noisy_blobs

    gen = np.random.default_rng(seed=seed)

    w = gen.standard_normal(size=(dim,1000))
    x = z @ w + gen.standard_normal(size=(z.shape[0],))*noise_sd_out

    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=test_size)

    return GeneralToyDset(train_x,train_y,transform=torch.from_numpy), GeneralToyDset(test_x,test_y,transform=torch.from_numpy)

class shapes3dDset(Dataset):

    def __init__(self,filepath,indices):

        self.file = h5py.File(filepath)
        self.n_images, self.H,self.W,self.C = self.file['images'].shape
        self.indices = indices
        self.n_used = len(self.indices)

    def __len__(self):

        return self.n_used 
    
    def __getitem__(self,index):

        idx = self.indices[index]
        image, label = self.file['images'][idx],self.file['labels'][idx]

        return (image.astype(np.float32)/255,label.astype(np.float32))
    


def get_3d_shapes(dpath,seed,test_size=0.2):

    dfile = os.path.join(dpath,'3dshapes.h5')
    dataset = h5py.File(dfile,'r')
    #images,labels = np.asarray(dataset['images']).astype(np.float32),np.asarray(dataset['labels'])
    (B,H,W,C) = dataset['images'].shape
    #images /= 255
    #images = np.swapaxes(images.astype(np.float32),axis1=1,axis2=3) # B C W H
    #images = np.swapaxes(images.astype(np.float32),axis1=2,axis2=3) # B C H W
    
    gen = np.random.default_rng(seed=seed)

    order = gen.choice(B,B,replace=False)
    train_end = int(round(B * (1-test_size)))

    #train_x,train_y = images[order[:train_end]],labels[order[:train_end]]
    #transform = lambda x: torch.from_numpy(x).permute(2,0,1)

    #test_x,test_y = images[order[train_end:]],labels[order[train_end:]]
    
    return shapes3dDset(dfile,order[:train_end]), shapes3dDset(dfile,order[train_end:])


def get_index(factors):
  """ from the 3dShapes Github
  Converts factors to indices in range(num_data)
  Args:
    factors: np array shape [6,batch_size].
             factors[i]=factors[i,:] takes integer values in 
             range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

  Returns:
    indices: np array shape [batch_size].
  """
  indices = 0
  base = 1
  for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
    indices += factors[factor] * base
    base *= _NUM_VALUES_PER_FACTOR[name]
  return indices

def get_3d_shapes_fixed_factors(dpath,seed,fixed_factors,test_size=0.2):


    gen = np.random.default_rng(seed=seed)
    dfile = os.path.join(dpath,'3dshapes.h5')
    dataset = h5py.File(dfile,'r')
    #images,labels = np.asarray(dataset['images']).astype(np.float32),np.asarray(dataset['labels'])
    (B,H,W,C) = dataset['images'].shape

    ### turn this into the same factor value for each run -- that way we can actually make meaningful plots
    fixed_factor_values = [int(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[f]]//2) for  f in fixed_factors]

    valid_indices = get_factor_indices(fixed_factors,fixed_factor_values)
    B = len(valid_indices)
    order = gen.choice(B,B,replace=False)
    train_end = int(round(B * (1-test_size)))

    return shapes3dDset(dfile,valid_indices[order[:train_end]]), shapes3dDset(dfile,valid_indices[order[train_end:]])

def get_factor_indices(fixed_factors,fixed_factor_values):

    """
    expects fixed factors and factor values to be in the same order.
    also expects them to be in numerical order (factor 1 comes before factor 2, etc.)
    """
    assert len(fixed_factors) == len(fixed_factor_values), print("each fixed factor must have a fixed value!")

    n_indices = np.prod([_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[ii]] for ii in range(6) if ii not in fixed_factors])
    
    print(f"there should be {n_indices} indices by the end of this function")

    factor_inds = []
    fixed_index = 0
    for factor,name in enumerate(_FACTORS_IN_ORDER):
        if factor in fixed_factors:
            factor_inds.append([fixed_factor_values[fixed_index]])
            fixed_index += 1
        else:
            factor_inds.append(list(range(_NUM_VALUES_PER_FACTOR[name])))

    factor_grid = np.meshgrid(*factor_inds)
    factor_grid = np.stack([f.flatten() for f in factor_grid],axis=0)
    assert factor_grid.shape[0] == 6, print(factor_grid.shape)
    assert factor_grid.shape[1] == n_indices,print(factor_grid.shape)

    inds = get_index(factor_grid)
    assert len(inds) == n_indices,print(n_indices.shape)

    return inds