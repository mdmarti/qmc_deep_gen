from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os,glob
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm 

def load_segmented_sylls(bird_filepath,sylls,test_size=0.2,seed=92):

    spec_files = []
    syll_ids = []
    for syll in sylls:
        sub_path = os.path.join(bird_filepath,f'syll_specs_{syll}/*')
        
        syll_files = glob.glob(os.path.join(sub_path,'*.hdf5'))
        spec_files += syll_files
        syll_ids += [syll]*len(syll_files)
        
    train_files,test_files,train_ids,test_ids = train_test_split(spec_files,syll_ids,test_size=test_size,random_state=seed)
    
    return (train_files,test_files),(train_ids,test_ids)
        

class bird_data(Dataset):

    def __init__(self,filenames,syll_ids,specs_per_file=20,transform=transforms.ToTensor(),
                 conditional=False,conditional_factor='fm'):


        self.filenames=filenames
        self.syll_ids = syll_ids
        self.specs_per_file = specs_per_file
        self.transform = transform
        self.conditional=conditional
        self.conditional_factor = conditional_factor

    def __len__(self):
        return len(self.filenames) * self.specs_per_file


    def __getitem__(self,index):

        load_index = index//self.specs_per_file
        spec_index = index%self.specs_per_file
        load_fn = self.filenames[load_index]
        syll_id = self.syll_ids[load_index]
        
        with h5py.File(load_fn,'r',locking=False) as f:
            spec = f['specs'][spec_index]

            if self.conditional:
                if self.conditional_factor == 'fm':
                    c = calc_fm(spec)
                elif self.conditional_factor == 'entropy':
                    c = calc_ent(spec)
                elif self.conditional_factor =='length':
                    c = f['offsets'][spec_index] - f['onsets'][spec_index]
                elif self.conditional_factor == 'locations':
                    ### this should ONLY be used for analysis and NOT for training
        
                    c = f['locations'][spec_index].decode('ASCII')
                elif self.conditional_factor == 'file':
                    ### this should ALSO only be used for analysis and NOT for training
                    c = f['audio_filenames'][spec_index].decode('ASCII')
                else:
                    raise NotImplementedError
        
        
        spec = self.transform(spec)

        if self.conditional:
            return (spec,c,syll_id)
        return (spec,syll_id)

class hdf5_data_general(Dataset):

    def __init__(self,filenames,syll_ids,transform=transforms.ToTensor(),
                 conditional=False,conditional_factor='fm'):
        
        self.filenames=filenames
        self.syll_ids = syll_ids
        self.transform = transform
        self.conditional=conditional
        self.conditional_factor = conditional_factor
        total, file_lens = self._get_len()

        self.length = total
        self.cumulative_file_nums = np.cumsum(file_lens).astype(np.int32)

    def _get_len(self):
       
        file_lens = []
        for fn in self.filenames:
            with h5py.File(fn,'r',locking=False) as f:
                file_lens.append(f['num_specs'])

        total_len = np.sum(file_lens)

        return total_len,file_lens

    def __len__(self):
        return self.length


    def __getitem__(self,index):

        load_index = np.argwhere(self.cumulative_file_nums >= index)[0].squeeze()
        spec_index = index - load_index
        load_fn = self.filenames[load_index]
        syll_id = self.syll_ids[load_index]
        
        with h5py.File(load_fn,'r',locking=False) as f:
            spec = f['specs'][spec_index]

            if self.conditional:
                if self.conditional_factor == 'fm':
                    c = calc_fm(spec)
                elif self.conditional_factor == 'entropy':
                    c = calc_ent(spec)
                elif self.conditional_factor =='length':
                    c = f['offsets'][spec_index] - f['onsets'][spec_index]
                elif self.conditional_factor == 'locations':
                    ### this should ONLY be used for analysis and NOT for training
        
                    c = f['locations'][spec_index].decode('ASCII')
                elif self.conditional_factor == 'file':
                    ### this should ALSO only be used for analysis and NOT for training
                    c = f['audio_filenames'][spec_index].decode('ASCII')
                else:
                    raise NotImplementedError
        
        
        spec = self.transform(spec)

        if self.conditional:
            return (spec,c,syll_id)
        return (spec,syll_id)
  
def load_gerbils(gerbil_filepath,families=[2],test_size=0.2,seed=92,check=True):

    specs_per_file = 100
    try:
        len(families)
    except:
        families = [families]
    specs_in_file = []
    all_family_specs= {f:[] for f in families}
    all_family_ids = {f:[] for f in families}

    for ii,family in enumerate(families):
        print(f"loading family{family}")
        spec_dir = os.path.join(gerbil_filepath,'processed-data',f"family{family}")
        spec_fns = glob.glob(os.path.join(spec_dir,'*.hdf5'))
        all_family_specs[family] += spec_fns
        
        all_family_ids[family].append(family*np.ones((len(spec_fns),)))
        
        if check:
            for spec_fn in tqdm(spec_fns,total=len(spec_fns)):
                with h5py.File(spec_fn,'r') as f:
                    sif = len(f['specs'])
                    specs_in_file.append(sif)

    if check:
        num_specs = np.unique(specs_in_file)
        assert len(num_specs) == 1, print(f"Files have different numbers of specs in them! {num_specs}")
        if num_specs[0] != specs_per_file:
            print(f"expected {specs_per_file} specs per file, found {num_specs[0]}; updating")
            specs_per_file = num_specs[0]
    #all_family_ids = np.hstack(all_family_ids)
    #assert num_specs[0] == specs_per_file,print("num_specs,specs_per_file)
    if test_size > 0:
        train_fns,test_fns,train_ids,test_ids = [],[],[],[]
        for family in families:
            specs,ids = all_family_specs[family],np.hstack(all_family_ids[family])
            tr_fn,te_fn,tr_id,te_id = train_test_split(specs,ids,test_size=test_size,random_state=seed)
            train_fns.append(tr_fn)
            test_fns.append(te_fn)
            train_ids.append(tr_id)
            test_ids.append(te_id)
        #train_fns,test_fns,train_ids,test_ids = train_test_split(all_family_specs,all_family_ids,test_size=test_size,random_state=seed)
        train_fns = sum(train_fns,[])
        test_fns = sum(test_fns,[])
        train_ids = np.hstack(train_ids)
        test_ids = np.hstack(test_ids)
    else:
        train_fns,test_fns = all_family_specs, all_family_specs
        train_ids,test_ids = all_family_ids,all_family_ids
    #train_ids = np.zeros((len(train_fns,)))
    #test_ids = np.zeros((len(test_fns,)))

    return (train_fns,test_fns),(train_ids,test_ids),specs_per_file

#### song features from syllables

def calc_ent(spec):
    """
    entropy...aka spectral flatness
    geometric mean of spectrum divided by arithmetic mean

    spec: Freq x Time image

    returns mean entropy/spectral flatness over the vocalization
    """

    
    (N,T) = spec.shape
    denom = np.sum(spec,axis=0,keepdims=True)/N #+1e-10)
    #valid_inds = np.argwhere(denom > 0) # length T indices
    numerator = np.exp(np.sum(np.log(spec + 1e-10),axis=0)/N)
    
    ent = numerator/denom

    weights = (denom > 0).astype(np.float32)
    weights /= np.sum(weights)
    return (ent*weights.squeeze()).sum() #np.nanmean(ent)



def calc_fm(spec):
    """
    spec should be h x w bins

    returns mean FM over the vocalization
    
    """
    dt = np.diff(spec,axis=1)
    df = np.diff(spec,axis=0)
    dt2 = np.amax(dt**2,axis=0)
    df2 = np.amax(df**2,axis=0)
    fm =np.arctan2(dt2,df2[:-1])/np.pi # divide by pi to make sure this is between -1,1
    weights = (np.sum(spec,axis=0) > 0).astype(np.float32)[:-1]
    weights /= np.sum(weights)
    return (fm * weights).sum()

def calc_duration(spec):

    """
    will give duration in terms of % of spectrogram (max length) with any detected sound
    """

    (N,T) = spec.shape
    sounds = np.sum(spec,axis=0)
    total_bins = (sounds > 0).astype(np.float32).sum()

    return total_bins/T

def calc_mean_freq(spec):

    """
    calculates SAP definition of mean frequency:
    mean of the squared spectral derivative

    returns average mean frequency over the vocalization, ranging from 0 (min freq) to 1 (max freq)
    """

    (N,T) = spec.shape


    dt = np.diff(spec,axis=1)
    df = np.diff(spec,axis=0)

    freq_weights = dt[:-1,:]**2 + df[:,:-1]**2
    time_weights = (np.sum(spec,axis=0) > 0).astype(np.float32)[:-1]
    time_weights /= np.sum(time_weights)

    freqs = np.linspace(0,N-1)[:,None]

    mean_freq = np.sum(freqs * freq_weights,axis=0)/np.sum(freq_weights,axis=0)

    return(mean_freq*time_weights).sum()




