import torch
from torch.utils.data import DataLoader,Dataset
from data.bird_data import *
from data.toy_dsets import *
from torchvision import transforms, datasets


def load_data(dataset_name,dataset_loc):

    n_workers = len(os.sched_getaffinity(0))

    if dataset_name.lower() == 'mnist':

        print("loading data...")
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(dataset_loc, train=True, download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True,num_workers=n_workers)
        test_data = datasets.MNIST(dataset_loc, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False,num_workers=n_workers)
        
        print("done!")

    elif dataset_name.lower() == 'celeba':

        print('loading data...')
        train_ims = torch.load(os.path.join(dataset_loc,'train_80x80.pt'))
        test_ims = torch.load(os.path.join(dataset_loc,'test80x80.pt'))  
        print('done!')
        train_data = CelebADsetIms(train_ims)
        test_data = CelebADsetIms(test_ims)
        train_loader = DataLoader(train_data,batch_size=64,num_workers=n_workers,shuffle=True)
        test_loader = DataLoader(test_data,batch_size=1,num_workers=n_workers,shuffle=False)
 

    elif dataset_name.lower() == 'finch':


        (train_files,test_files),(train_ids,test_ids) = load_segmented_sylls(dataset_loc,sylls=['A','B','C','D','D2','E'])
        train_ds = bird_data(train_files,train_ids)
        test_ds = bird_data(test_files,test_ids)
        train_loader = DataLoader(train_ds,num_workers=n_workers,shuffle=True,batch_size=64)
        test_loader = DataLoader(test_ds,num_workers=n_workers,shuffle=False,batch_size=1)

    return train_loader,test_loader