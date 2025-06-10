from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os,glob
import h5py
from sklearn.model_selection import train_test_split

def load_segmented_sylls(bird_filepath,sylls,test_size=0.2):

    spec_files = []
    syll_ids = []
    for syll in sylls:
        sub_path = os.path.join(bird_filepath,f'syll_specs_{syll}/*')
        
        syll_files = glob.glob(os.path.join(sub_path,'*.hdf5'))
        spec_files += syll_files
        syll_ids += [syll]*len(syll_files)
        
    train_files,test_files,train_ids,test_ids = train_test_split(spec_files,syll_ids,test_size=test_size)
    
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
        
