import torch
from torch.utils.data import DataLoader,Dataset
from data.bird_data import *
from data.toy_dsets import *
from torchvision import transforms, datasets
from data.mocap import *


def load_data(dataset_name,dataset_loc,batch_size=256,subj='54',frames_per_sample=1):

    n_workers = len(os.sched_getaffinity(0))

    if 'mnist' in dataset_name.lower():

        print("loading mnist...")
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(dataset_loc, train=True, download=True, transform=transform)
        #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=n_workers)
        test_data = datasets.MNIST(dataset_loc, train=False, download=True, transform=transform)
        #test_loader = DataLoader(test_data, batch_size=batch_size//4, shuffle=False,num_workers=n_workers)
        
    elif 'celeba' in dataset_name.lower():

        print('loading celeba...')
        train_ims = torch.load(os.path.join(dataset_loc,'train_80x80.pt'))
        test_ims = torch.load(os.path.join(dataset_loc,'test80x80.pt'))  

        train_data = CelebADsetIms(train_ims)
        test_data = CelebADsetIms(test_ims)
        #train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=n_workers,shuffle=True)
        #test_loader = DataLoader(test_data,batch_size=batch_size//4,num_workers=n_workers,shuffle=False)
 

    elif 'finch' in dataset_name.lower():

        print("loading bird....")

        (train_files,test_files),(train_ids,test_ids) = load_segmented_sylls(dataset_loc,sylls=['A','B','C','D','D2','E'])
        train_data = bird_data(train_files,train_ids)
        test_data = bird_data(test_files,test_ids)
        #train_loader = DataLoader(train_data,num_workers=n_workers,shuffle=True,batch_size=batch_size)
        #test_loader = DataLoader(test_data,num_workers=n_workers,shuffle=False,batch_size=batch_size//4)

    elif 'mocap' in dataset_name.lower():

        print('loading mocap...')

        (train_trials,test_trials),(train_labels,test_labels),(train_frames,test_frames),means,keys,motions,joints = get_samples(dataset_loc,subj,frames_per_sample)

        train_data = MocapDataset(train_trials,train_labels,means,motions,train_frames,joints,keys)
        test_data = MocapDataset(test_trials,test_labels,means,motions,test_frames,joints,keys)

    train_loader = DataLoader(train_data,num_workers=n_workers,shuffle=True,batch_size=batch_size)
    test_loader = DataLoader(test_data,num_workers=n_workers,shuffle=False,batch_size=batch_size//4)

    print("done!")

    return train_loader,test_loader