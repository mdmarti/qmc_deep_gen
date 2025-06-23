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
import json


latent_dim = 2

def make_qmc_model(latent_dim,device):
    
    decoder_qmc = nn.Sequential(
            TorusBasis(),
            nn.Linear(2*latent_dim,2048),
            #nn.ReLU(),
            #nn.Linear(2048, 32*7*7),
            nn.ReLU(),
            nn.Linear(2048,64*7*7),
            nn.Unflatten(1, (64, 7, 7)),
            ResCellNVAESimple(64,expand_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1,groups=64), #nn.Linear(64*7*7,32*14*14),
            nn.Conv2d(64,32,1),
            ResCellNVAESimple(32,expand_factor=4),#nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1,groups=32),#nn.Linear(32*14*14,1*28*28),
            nn.Conv2d(32,16,1),
            ResCellNVAESimple(16,expand_factor=4),
            ResCellNVAESimple(16,expand_factor=2),
            ResCellNVAESimple(16,expand_factor=1),
            nn.Conv2d(16,1,1),
            nn.Sigmoid(),
        )
    qmc_model = QMCLVM(latent_dim=2,device=device,decoder=decoder_qmc)
    
    return qmc_model

def generate_test_evidence(lattices,model,loader,evidence,mc_func):

    qe,rqe,me =[],[],[]
    all_labels=[]
    for lattice in lattices:
        n_points = len(lattice)

        qmce,rqmce,mce = [],[],[]
        labels=[]
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        for batch in tqdm(loader,total=len(loader)):
            

            (data,ls) = batch
            labels.append(ls)
            #data = data
            mc_pts = mc_func(n_points)
            with torch.no_grad():
                qmc_samples = model(lattice.to(model.device),random=False,mod=True).detach().cpu()
                rqmc_samples = model(lattice.to(model.device),random=True,mod=True).detach().cpu()
                mc_samples = model(mc_pts.to(model.device),random=False,mod=False).detach().cpu()
                qmc_evidence= evidence(qmc_samples,data).numpy()
                rqmc_evidence = evidence(rqmc_samples,data).numpy()
                mc_evidence = evidence(mc_samples,data).numpy()

            qmce.append(qmc_evidence)
            rqmce.append(rqmc_evidence)
            mce.append(mc_evidence)
            labels.append(ls.detach().cpu().numpy())

        all_labels.append(np.hstack(labels))
        qe.append(np.hstack(qmce))
        rqe.append(np.hstack(rqmce))
        me.append(np.hstack(mce))

    return qe,rqe,me,all_labels


def mc_unif(n_points,dim):


    return torch.rand(n_points,dim,dtype=torch.float32)

def evidence_plot(qmc_est,rqmc_est,mc_est,n_lattice_points,title,save_loc='ev_plot.png',n_plot=35):

    ax = plt.gca()
    ii = 0
    labels = ['QMC Evidence estimate','RQMC Evidence estimate', 'MC Evidence estimate']
    lines = []
    plot_order = np.random.choice(len(qmc_est[0]),n_plot,replace=False).squeeze()
    for n,q,r,m in zip(n_lattice_points,qmc_est,rqmc_est,mc_est):

        g1 = ax.scatter([n]*n_plot+np.random.randn(len(q))*.05,q[plot_order],s=3,color='tab:blue')
        g2 = ax.scatter([n]*n_plot+np.random.randn(len(q))*.05,r[plot_order],s=2,color='tab:orange')
        g3 = ax.scatter([n]*n_plot+np.random.randn(len(q))*.05,m[plot_order],s=1, color='tab:green')
        if ii == 0:
            lines.append(g1)
            lines.append(g2)
            lines.append(g3)

    ax.set_ylabel('Negative marginal log likelihood')
    ax.set_xlabel("Number of probe lattice points")

    ax.spines[['right','top']].set_visible(False)
    ax.legend(lines,labels,frameon=False)
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(save_loc)
    plt.close()


def run_qmc_mc_comparison_experiments(save_location,dataloc,nEpochs=300,train_mode='rqmc'):


    ############ shared model setup ###############################
    n_workers = len(os.sched_getaffinity(0))
    #save_location += train_mode
    #############################
    print("loading data...")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(dataloc, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True,num_workers=n_workers)
    test_data = datasets.MNIST(dataloc, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False,num_workers=n_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fib_number = [7,11,12,16,21]
    n_lattice_points = [fib(m) for m in fib_number]
    qmc_latent_dim=2
    print("done!")

    if train_mode == 'rqmc':
        random = True 
    else:
        random=False
    ############## Set up qmc model, training ###################################

    mc_fnc = lambda n: mc_unif(n,2)
    test_base_sequences = [gen_fib_basis(m) for m in fib_number]
    for n_train,f_num in zip(n_lattice_points,fib_number):
        print(f"now training on {f_num},{n_train} points")
        train_base_sequence = gen_fib_basis(m=f_num)
        
        qmc_loss_function = lambda x,y: binary_evidence(x,y,reduce=True,batch_size=1000)

        qmc_model = make_qmc_model(qmc_latent_dim,device=device)
        save_qmc = os.path.join(save_location,f'rqmc_{n_train}_points_train_{train_mode}.tar')
        if not os.path.isfile(save_qmc):
            print("now training qmc model")
            ## starting this run takes a little while for some reason...
            if train_mode == 'mc':
                qmc_model,qmc_opt,qmc_losses,labels = train_qmc.train_loop_mc(qmc_model,train_loader,
                                                                          qmc_loss_function,mc_fnc,
                                                                           nEpochs=nEpochs)
            else:
                qmc_model,qmc_opt,qmc_losses,labels = train_qmc.train_loop(qmc_model,train_loader,
                                                                           train_base_sequence.to(device),qmc_loss_function,
                                                                           nEpochs=nEpochs,random=random)
            save(qmc_model.to('cpu'),qmc_opt,qmc_losses,fn=save_qmc)
            qmc_model.to(device)
            qmc_losses = np.array(qmc_losses)
            ax = plt.gca()
            ax.plot(-np.array(qmc_losses))
            ax =  format_plot_axis(ax,ylabel='log evidence',xlabel='update number',xticks=ax.get_xticks(),yticks=ax.get_yticks())
            plt.savefig(os.path.join(save_location,f'qmc_{n_train}_points_{train_mode}_train_stats.svg'))
            plt.close()
            
        else:
            qmc_opt = Adam(qmc_model.parameters(),lr=1e-3)
            qmc_model,qmc_opt,qmc_losses = load(qmc_model,qmc_opt,save_qmc)
        
        stats_save_path = os.path.join(save_location,f"model_evidence_{n_train}_points_{train_mode}_values.json")
        ev_fnc = lambda x,y: binary_evidence(x,y,reduce=False,batch_size=fib(12))
        if not os.path.isfile(stats_save_path):
            qmce,rqmce,mce = generate_test_evidence(test_base_sequences,qmc_model,test_loader,ev_fnc,mc_fnc)
            ev_stats = {'qmc': [q.tolist() for q in qmce],'rqmc':[r.tolist() for r in rqmce],'mc':[m.tolist() for m in mce]}
            with open(stats_save_path,'w') as f:
                json.dump(ev_stats,f)
        else:
            with open(stats_save_path,'r') as f:
                ev_stats = json.load(f)
            qmce,rqmce,mce =  [np.array(q) for q in ev_stats['qmc']],[np.array(r) for r in ev_stats['rqmc']],[np.array(m) for m in ev_stats['mc']]   
        ev_plot_save_fn = os.path.join(save_location,f'model_evidence_comparison_{n_train}_points_{train_mode}_train.png')
        evidence_plot(qmce,rqmce,mce,n_lattice_points,title=f"Model evidence estimates, trained with {n_train} lattice points",save_loc=ev_plot_save_fn)
    
if __name__ == '__main__':

    fire.Fire(run_qmc_mc_comparison_experiments)


