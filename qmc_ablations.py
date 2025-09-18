from models.sampling import *
from models.qmc_base import * 
from plotting.visualize import *
from train.losses import *
from data.bird_data import *
from models.vae_base import *
import train.train as train_qmc
from tqdm import tqdm
from models.layers import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import fire
from torch.optim import Adam
from train.model_saving_loading import *
from data.utils import load_data
import json
from models.utils import get_decoder_arch

latent_dim = 2

def mc_unif(n_points,dim):


    return torch.rand(n_points,dim,dtype=torch.float32)

def run_qmc_mc_comparison_experiments(save_location,dataloc,dataset,batch_size=128,
                            nEpochs=300,rerun=False,train_lattice_m=15,
                            test_lattice_m=18,
                            frames_per_sample=1,
                            var=0.1,families=[2],
                            n_iters=10):


    ############ shared model setup ###############################
    
    #save_location += train_mode
    #############################
    print("loading data...")
    



    ################ Shared Setup ######################################
    #n_workers = len(os.sched_getaffinity(0))

    save_location = os.path.join(save_location,dataset)
    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    print(f"Training on {dataset} data")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qmc_latent_dim=2#3 if (('celeba' in dataset.lower()) or ('shapes3d' in dataset.lower())) else 2

    if 'finch' in dataset.lower():
        cm = 'viridis'
        origin = 'lower'
    elif 'gerbil' in dataset.lower():
        cm = 'inferno'
        origin = 'lower'
    else:
        cm = 'gray'
        origin = None

    if qmc_latent_dim == 2:
        train_loader,test_loader = load_data(dataset,dataloc,batch_size=batch_size,frames_per_sample=frames_per_sample,
                                             families=families)
        train_lattice = gen_fib_basis(m=train_lattice_m)
        test_lattice = gen_fib_basis(m=test_lattice_m)
        n_samples_train= len(train_lattice)
        n_samples_test = len(test_lattice)
        mc_fnc_train = lambda: mc_unif(n_samples_train,2)
        mc_fnc_test = lambda: mc_unif(n_samples_test,2)
    

    rqmc_save_loc = os.path.join(save_location,f'rqmc_test_losses_{dataset}.json')
    nonperiodic_save_loc = os.path.join(save_location,f'qmc_test_losses_{dataset}.json')
    gaussian_save_loc = os.path.join(save_location,f'mc_test_losses_{dataset}.json')

    ############## Set up rqmc model, training ###################################
    rqmc_test_losses = []
    if not os.path.isfile(rqmc_save_loc):
        for iter in range(n_iters):
            print("*"*25)
            print(f"Now evaluating rqmc {iter}")
            print('*'*25)
            qmc_save_path = os.path.join(save_location,f'rqmc_{iter}_2d_mnist_train.tar')
            qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim,n_per_sample=frames_per_sample)
            qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

            qmc_loss_func = binary_evidence if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda samples,data: gaussian_evidence(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
            qmc_lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 

            if not os.path.isfile(qmc_save_path):
                qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop(qmc_model,train_loader,train_lattice.to(device),qmc_loss_func,\
                                                                    nEpochs=nEpochs,
                                                                    verbose=('celeba' in dataset.lower()) or ('shapes3d' in dataset.lower()))
                print("Done training!")
                qmc_model.eval()
                with torch.no_grad():
                    qmc_model.eval()
                    qmc_test_losses = train_qmc.test_epoch(qmc_model,test_loader,test_lattice.to(device),qmc_loss_func)
                qmc_run_info = {'train':qmc_losses,'test':qmc_test_losses}
                save(qmc_model.to('cpu'),qmc_opt,qmc_run_info,fn=qmc_save_path)
                qmc_model.to(device)
                

            else:
                qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
                qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_save_path)
                qmc_model.eval()
                qmc_losses,qmc_test_losses = qmc_run_info['train'],qmc_run_info['test']
                qmc_model.to(device)

            rqmc_test_losses.append(np.nanmean(qmc_test_losses))
        save_data = {'test_losses': rqmc_test_losses}
        with open(rqmc_save_loc,'w') as f:
                json.dump(save_data,f)
    else:
        print("already trained & evaled! passing")

    ############## Set up qmc model, training ###################################
    nonperiodic_test_losses = []
    if not os.path.isfile(nonperiodic_save_loc):
        for iter in range(n_iters):
            print("*"*25)
            print(f"Now evaluating qmc {iter}")
            print('*'*25)
            qmc_save_path = os.path.join(save_location,f'qmc_{iter}_2d_mnist_train.tar')
            qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim,n_per_sample=frames_per_sample)
            qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

            qmc_loss_func = binary_evidence if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda samples,data: gaussian_evidence(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
            qmc_lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 

            if not os.path.isfile(qmc_save_path):
                qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop(qmc_model,train_loader,train_lattice.to(device),qmc_loss_func,\
                                                                    nEpochs=nEpochs,
                                                                    verbose=('celeba' in dataset.lower()) or ('shapes3d' in dataset.lower()),
                                                                    random=False)
                print("Done training!")
                qmc_model.eval()
                with torch.no_grad():
                    qmc_model.eval()
                    qmc_test_loss = train_qmc.test_epoch(qmc_model,test_loader,test_lattice.to(device),qmc_loss_func)
                qmc_run_info = {'train':qmc_losses,'test':qmc_test_loss}
                save(qmc_model.to('cpu'),qmc_opt,qmc_run_info,fn=qmc_save_path)
                qmc_model.to(device)
                

            else:
                qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
                qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_save_path)
                qmc_model.eval()
                qmc_losses,qmc_test_loss = qmc_run_info['train'],qmc_run_info['test']
                qmc_model.to(device)

            qmc_test_losses.append(np.nanmean(qmc_test_loss))
        save_data = {'test_losses':qmc_test_losses}
        with open(nonperiodic_save_loc,'w') as f:
                json.dump(save_data,f)
    else:
        print("already trained & evaled! passing")

    ############## Set up mc model, training ###################################
    gaussian_test_losses = []
    if not os.path.isfile(gaussian_save_loc):
        for iter in range(n_iters):
            print("*"*25)
            print(f"Now evaluating mc {iter}")
            print('*'*25)
            qmc_save_path = os.path.join(save_location,f'mc_{iter}_2d_mnist_train.tar')
            qmc_decoder = get_decoder_arch(dataset_name=dataset,latent_dim=qmc_latent_dim,n_per_sample=frames_per_sample)
            qmc_model = QMCLVM(latent_dim=qmc_latent_dim,device=device,decoder=qmc_decoder)

            qmc_loss_func = binary_evidence if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower()) else lambda samples,data: gaussian_evidence(samples,data,var=var) #or ('gerbil' in dataset.lower()) 
            qmc_lp = binary_lp if ('mnist' in dataset.lower()) or ('gerbil' in dataset.lower())  else lambda samples,data: gaussian_lp(samples,data,var=var) #or ('gerbil' in dataset.lower()) 

            if not os.path.isfile(qmc_save_path):
                qmc_model,qmc_opt,qmc_losses = train_qmc.train_loop_mc(qmc_model,train_loader,
                                                                      qmc_loss_func,mc_fnc_train,\
                                                                    nEpochs=nEpochs)
                print("Done training!")
                qmc_model.eval()
                with torch.no_grad():
                    qmc_model.eval()
                    qmc_test_losses = train_qmc.test_epoch(qmc_model,test_loader,test_lattice.to(device),qmc_loss_func)
                qmc_run_info = {'train':qmc_losses,'test':qmc_test_losses}
                save(qmc_model.to('cpu'),qmc_opt,qmc_run_info,fn=qmc_save_path)
                qmc_model.to(device)
                

            else:
                qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
                qmc_model,qmc_opt,qmc_run_info = load(qmc_model,qmc_opt,qmc_save_path)
                qmc_model.eval()
                qmc_losses,qmc_test_losses = qmc_run_info['train'],qmc_run_info['test']
                qmc_model.to(device)

            mc_test_losses.append(np.nanmean(qmc_test_losses))
        save_data = {'test_losses': mc_test_losses}
        with open(gaussian_save_loc,'w') as f:
                json.dump(save_data,f)
    else:
        print("already trained & evaled! passing")
        


if __name__ == '__main__':

    fire.Fire(run_qmc_mc_comparison_experiments)


