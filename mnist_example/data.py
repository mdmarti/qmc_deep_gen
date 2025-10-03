from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import os

def load_mnist(dataset_loc,max_workers=4,batch_size=128):


    n_workers = min(len(os.sched_getaffinity(0)),max_workers)
    print("loading mnist...")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(dataset_loc, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(dataset_loc, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data,num_workers=n_workers,shuffle=True,batch_size=batch_size)
    test_loader = DataLoader(test_data,num_workers=n_workers,shuffle=False,batch_size=batch_size)

    return train_loader,test_loader