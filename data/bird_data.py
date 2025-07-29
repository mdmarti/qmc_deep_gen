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

    def __init__(self,filenames,syll_ids,specs_per_file=20,transform=transforms.ToTensor()):


        self.filenames=filenames
        self.syll_ids = syll_ids
        self.specs_per_file = specs_per_file
        self.transform = transform

    def __len__(self):
        return len(self.filenames) * self.specs_per_file


    def __getitem__(self,index):

        load_index = index//self.specs_per_file
        spec_index = index%self.specs_per_file
        load_fn = self.filenames[load_index]
        syll_id = self.syll_ids[load_index]
        
        with h5py.File(load_fn,'r') as f:
            spec = f['specs'][spec_index]
        spec = self.transform(spec)

        return (spec,syll_id)
    
def load_gerbils(gerbil_filepath,specs_per_file,families=[2],test_size=0.2,seed=92):

    try:
        len(families)
    except:
        families = [families]
    specs_in_file = []
    all_family_specs= []
    all_family_ids = []
    for ii,family in enumerate(families):
        print(f"loading family{family}")
        spec_dir = os.path.join(gerbil_filepath,'processed-data',f"family{family}")
        spec_fns = glob.glob(os.path.join(spec_dir,'*.hdf5'))
        all_family_specs += spec_fns
        all_family_ids.append(ii*np.ones((len(spec_fns),)))

        for spec_fn in tqdm(spec_fns,total=len(spec_fns)):
            with h5py.File(spec_fn,'r') as f:
                sif = len(f['specs'])
                specs_in_file.append(sif)

    num_specs = np.unique(specs_in_file)
    assert len(num_specs) == 1, print(f"Files have different numbers of specs in them! {num_specs}")
    if num_specs[0] != specs_per_file:
        print(f"expected {specs_per_file} specs per file, found {num_specs[0]}; updating")
        specs_per_file = num_specs[0]
    all_family_ids = np.hstack(all_family_ids)
    #assert num_specs[0] == specs_per_file,print("num_specs,specs_per_file)

    train_fns,test_fns,train_ids,test_ids = train_test_split(all_family_specs,all_family_ids,test_size=test_size,random_state=seed)
    #train_ids = np.zeros((len(train_fns,)))
    #test_ids = np.zeros((len(test_fns,)))

    return (train_fns,test_fns),(train_ids,test_ids),specs_per_file
