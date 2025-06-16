import torch
from torch.utils.data import Dataset,DataLoader

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