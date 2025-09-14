import torch
import os
from models.sampling import gen_fib_basis
from models.utils import *
import fire
from models.qmc_base import QMCLVM
from train.losses import *
from train.model_saving_loading import *
from analysis.model_helpers import get_stacked_posterior
from tqdm import tqdm
from data.toy_dsets import *
from data.toy_dsets import _FACTORS_IN_ORDER,_NUM_VALUES_PER_FACTOR
from torch.utils.data import DataLoader
import json
from torch.optim import Adam

def embed_shapes_data(model_loc,data_loc,save_loc,batch_size=32):


    #### get one grid for EACH latent factor value!
    n_workers = len(os.sched_getaffinity(0))
    ### load qmc model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading shapes model from ", model_loc)
    decoder = get_decoder_arch(dataset_name='shapes3d',latent_dim=2)
    model = QMCLVM(latent_dim=2,device=device,decoder=decoder)
    opt = Adam(model.parameters(),lr=1e-3)
    lp = lambda target,recon: gaussian_lp(recon,target,var=0.1) #or ('gerbil' in dataset.lower())

    model,opt,run_info = load(model,opt,model_loc)
    model.eval()
    #train_lattice = gen_fib_basis(m=train_lattice_m)
    test_lattice = gen_fib_basis(m= 18).to(device)

    ### load shapes data
    dfile = os.path.join(data_loc,'3dshapes.h5')
    dataset = h5py.File(dfile,'r')
    #images,labels = np.asarray(dataset['images']).astype(np.float32),np.asarray(dataset['labels'])
    (B,H,W,C) = dataset['images'].shape
    posteriors = {}
    for ii,factor in enumerate(_FACTORS_IN_ORDER):
        posteriors[factor] = []
        for jj, value in enumerate(range(1,_NUM_VALUES_PER_FACTOR[factor]+1)):
            print(f"now gathering aggregated posterior for {factor}:{value}")
            inds = get_factor_indices(fixed_factors =[ii],fixed_factor_values=[jj])
            ds = shapes3dDset(dfile,inds)
            dl = DataLoader(ds,num_workers=n_workers,shuffle=False,batch_size=batch_size)

            stacked_posterior = get_stacked_posterior(model,test_lattice,dl,lp)
            posteriors[factor].append(stacked_posterior.mean(axis=0).tolist())


    save_file = os.path.join(save_loc,'all_factor_posteriors.json')
    with open(save_file,'w') as f:
        json.dump(posteriors,f)

if __name__ == '__main__':

    fire.Fire(embed_shapes_data)
    
