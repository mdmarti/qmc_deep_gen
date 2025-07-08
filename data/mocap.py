import third_party.amc_parser as amc
import numpy as np 
import os
import glob
from tqdm import tqdm 
import copy 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from scipy import stats
import matplotlib as mpl
from itertools import repeat 

def getRotation(joints,motion,excluded = ['toes','hand','fingers','thumb','hipjoint']):
    
    rotations = []
    globalrots = []
    jointKey = []
    #print(len(joints))
    #print(len(joints.keys()))
    for j in joints.keys():
        joint = joints[j]
        #print(j)
        drop = False
        for kk in excluded:
            if kk in joint.name:
                drop = True
                print(f'dropping {joint.name}')
                continue
        if not drop:
            if joint.name == 'root':

                globalRot = np.deg2rad(motion['root'][3:])

                
                jointKey.append((j,len(globalRot)))
                globalrots.append(np.hstack([np.sin(globalRot),np.cos(globalRot)])) # SHOULD be size 2xDoF
            else:
                idx = 0
                rotation = []
                for axis, lm in enumerate(joint.limits):
                    if not np.array_equal(lm, np.zeros(2)):
                        rotation.append(motion[joint.name][idx])
                        idx += 1
                    else:
                         rotation.append(0)
                    
                #print(joint.name)
                #if len(rotation) > 0:
                #rotation = np.hstack(rotation)
                #print(f"original values {joint.name}: {rotation}")

                rotation = np.deg2rad(rotation)

                jointKey.append((j,len(rotation)))
                rotations.append(np.hstack([np.sin(rotation),np.cos(rotation)]))

    return np.hstack(rotations),np.array(globalrots),jointKey

def preprocess_mocap_motion(joints,motion,n_frames_per_sample=3):

    ### only process one motion at a time. this is getting too complicated
    """
    this preprocessing follows the steps from "Gaussian Process Dynamical Models for Human Motion" (Wang, Fleet, & Hertzmann 2008)
    Briefly, we treat each pose a 44 Euler angles for joints, three global (torso) pose angles,
    and 3 global (torso) translational velocities.
    All data are then mean-subtracted.
    we treat each data point as a set of n frames over time, so we stack after mean subtracting
    
    SINCE we have the sin and cosine basis now, we shouldn't need to mean subtract....but test out just to be sure 
    """

    

    rotTrial,globalRotTrial = [],[]
    for frame in motion:
        #print(subject,frame)
        rot,globalrot,jointNames = getRotation(joints,frame,excluded=['nothing'])
        
        rotTrial.append(rot)
        globalRotTrial.append(globalrot)

    rotTrial = np.vstack(rotTrial)
    globalRots = np.vstack(globalRotTrial)
    globalTranslation = np.diff(globalRots,axis=0)
    globalTranslation = np.vstack([globalTranslation,globalTranslation[-1:,:]])
    allRots = np.hstack([globalRots,rotTrial,globalTranslation])
    muRot = np.nanmean(allRots,axis=0)
    trial_converted_orig = frames2samples(allRots,n_frames_per_sample)
    #subjOrig.append(trial_converted_orig)
    allRots = allRots# - muRot
    #subjMeans.append(muRot)

    trial_converted = frames2samples(allRots,n_frames_per_sample)
    #subjTrajs.append(trial_converted)
    #subjKeys.append(jointNames)

    #allTrajs = allTrajs + subjTrajs
    #traj_means = traj_means + subjMeans
    #conversion_keys = conversion_keys + subjKeys
    #origTrajs = origTrajs + subjOrig
    
    return trial_converted,muRot,jointNames
    
def frames2samples(frames,n_frames_per_sample):
    
    n_frames = len(frames)
    samples = []
    for start in range(0,n_frames - n_frames_per_sample):
    
        samples.append(frames[start:start+n_frames_per_sample])
    
    return np.stack(samples,axis=0)

##### TO DO: get samples (datapath,subject)
##### think about decoder arch for these data

		
def sample2motion(sample,motion,conversion_key,testing = False):

	"""
	converts a trajectory back into a motion, for plotting purposes.
	will replace the values for all joints with the model predictions.
	This assumes that trajectories have already been re-meaned.
	
	this also only takes one sample, one motion at a time -- and assumes the samples in the motion correspond to the samples in the sample!!
	"""

	#print(
	#traj = np.rad2deg(sample)
	newMotion = [] #copy.deepcopy(motion)
	
	for sampleFrame,motionFrame in zip(sample,motion):

		newFrame = copy.deepcopy(motionFrame)
		start_ind = 0
		for (jointName,DoF) in conversion_key:
			# these are added to the list in the same order that joints are added to the sample list, so this should be okay
			#print(f"replacing {jointName}")
			sins = sampleFrame[start_ind:start_ind+DoF]
			coss = sampleFrame[start_ind+DoF:start_ind+2*DoF]
			rads_recon = np.arctan2(sins,coss)
			degrees_recon = np.rad2deg(rads_recon)
			#degrees_recon[degrees_recon <0] = 360 + degrees_recon[degrees_recon < 0]
			if jointName == 'root':
				
				newFrame[jointName] = motionFrame[jointName][:3] + degrees_recon.tolist() # unsure if this will work, but it should
				if testing:
					assert np.all(np.isclose(np.array(motionFrame[jointName][3:]),degrees_recon)),print(motionFrame[jointName][3:],degrees_recon.tolist(),start_ind,start_ind+DoF)
			else:
				newFrame[jointName] = degrees_recon.tolist()
				if testing:
					assert np.all(np.isclose(degrees_recon,np.array(motionFrame[jointName]))),print(degrees_recon,np.array(motionFrame[jointName]),start_ind,start_ind+DoF)

			start_ind += 2*DoF
		newMotion.append(newFrame)

	return newMotion

def plot_sample(sample,joints,motion,conversion_key,n_per_sample=5):

	
    predicted_motion = sample2motion(sample,motion,conversion_key,testing=True)
    fig,axs = plt.subplots(nrows=1,ncols=n_per_sample,subplot_kw=dict(projection='3d'),figsize=(20,5))
        
    for ii,frame in enumerate(predicted_motion):
        ax = axs[ii]    
        #ax = fig.add_subplot(111,projection='3d')
        #ax = plt.gca()
        joints['root'].set_motion(frame)
        pts = joints['root'].to_dict()
    
        xs,ys,zs = [],[],[]
        ls = []
        for j in pts.values():
            xs.append(j.coordinate[0,0])
            ys.append(j.coordinate[1,0])
            zs.append(j.coordinate[2,0])
        ax.plot(zs,xs,ys,'b.')
        #print(l)
        #ls.append(l[0])
        ax.view_init(azim=90,elev=10)
        for iter,j in enumerate(pts.values()):
            child = j
            if child.parent is not None:
                parent = child.parent
                xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
                ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
                zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
                l = ax.plot(zs,xs,ys,'-r')
                ls.append(l[0])
    
        xlims,ylims,zlims = ax.get_xlim(),ax.get_ylim(),ax.get_zlim()
        ranges = [xlims[1]-xlims[0],ylims[1]-ylims[0],zlims[1]-zlims[0]]
        max_range = np.amax(ranges)
        padding = [max_range - ranges[0],max_range - ranges[1],max_range - ranges[2]]
        xlims = [xlims[0] - padding[0]//2,xlims[1] + padding[0]//2]
        ylims = [ylims[0] - padding[1]//2,ylims[1] + padding[1]//2]
        zlims = [zlims[0] - padding[2]//2,zlims[1] + padding[2]//2]
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
        
    
    
    plt.show()
    plt.close()
      
def get_samples(datapath,subject,n_frames_per_sample=4,test_size=0.2):
    
    
    asf = glob.glob(os.path.join(datapath,subject,'*.asf'))[0]
    amcs = glob.glob(os.path.join(datapath,subject,'*.amc'))
    file_numbers = [int(a.split('_')[-1].split('.amc')[0]) for a in amcs]
    order = np.argsort(file_numbers)
    amcs = [amcs[o] for o in order]
    
    frames = [amc.parse_amc(am) for am in amcs]
    joints = amc.parse_asf(asf) # for asf in asfs]
    
    trials,means,keys,frame_nos,labels = [],[], [],[],[]
    
    for ii,trial in tqdm(enumerate(frames),total=len(frames)):
        traj,mean,key = preprocess_mocap_motion(joints,trial,n_frames_per_sample=n_frames_per_sample)
    
        trials.append(traj)
        means.append(mean)
        keys.append(key)
        frame_nos.append(np.arange(len(traj)))
        labels.append(ii * np.ones((len(traj),),dtype=np.int32))
    
    trials,labels,frame_nos =np.vstack(trials),np.hstack(labels),np.hstack(frame_nos)
    train_trials,test_trials,train_labels,test_labels,train_frames,test_frames = train_test_split(trials,labels,frame_nos,test_size=test_size)
    
    
    return (train_trials,test_trials),(train_labels,test_labels),(train_frames,test_frames),means,keys,frames,joints

class MocapDataset(Dataset):

    def __init__(self,samples,labels,means,motions,frame_nos,joints,conversion_keys,transform=transforms.ToTensor()):
        """
        there should only ever be joints for one (1) individual
        
        put transformation to sin(theta), cos(theta) here?
        """

        self.samples = samples
        self.labels = labels
        self.means = means
        self.motions = motions
        self.frame_nos = frame_nos
        self.joints = joints
        self.conversion_keys = conversion_keys
        self.length = len(samples)
        self.n_per_sample=0
        self.transform=transform

    def __len__(self):

        return self.length

    def __getitem__(self,index,return_all_info=False):


        sample = self.samples[index]
        if self.n_per_sample == 0:
              self.n_per_sample = sample.shape[0]
        sample = self.transform(sample).unsqueeze(0)#,(1,sample.shape[0],sample.shape[1])))#.unsqueeze(0)
        label = self.labels[index]

        if return_all_info:

            frame_no = self.frame_nos[index]
            motion = self.motions[label]
            key = self.conversion_keys[label]
            mean = self.means[label]
            return (sample, label), (frame_no,motion,key,mean)

        
        return sample, label


def plot_motion_on_ax(ax,motion,joints):

    joints['root'].set_motion(motion)
    pts = joints['root'].to_dict()

    xs,ys,zs = [],[],[]
    ls = []
    for j in pts.values():
        xs.append(j.coordinate[0,0])
        ys.append(j.coordinate[1,0])
        zs.append(j.coordinate[2,0])
    ax.plot(zs,xs,ys,'b.')
    #print(l)
    #ls.append(l[0])
    ax.view_init(azim=90,elev=10)
    for iter,j in enumerate(pts.values()):
        child = j
        if child.parent is not None:
            parent = child.parent
            xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
            ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
            zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
            l = ax.plot(zs,xs,ys,'-r')
            ls.append(l[0])

    return ax

EPS1 = 1e-15
EPS2 = 1e-6
def model_grid_plot(model,n_samples_dim,base_motion,joints,conversion_key,fn='',show=True,model_type='qmc'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_samples_dim = 10
    n_samples=n_samples_dim**2
    cmap=mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(-1,n_samples)
    with torch.no_grad():
        #z = torch.rand(n_samples, 2).to(device)
        xx,yy = torch.meshgrid([torch.linspace(EPS1,1-EPS2,n_samples_dim)]*2,indexing='ij')
        z = torch.stack([xx.flatten(),yy.flatten()],axis=-1).to(device)
        if model_type == 'vae':

            dist = stats.norm
            z = torch.from_numpy(dist.ppf(z.detach().cpu().numpy())).to(torch.float32).to(model.device)
        sample = model.decoder(z).detach().cpu().numpy().squeeze()

    sample = np.nanmean(sample,axis=1)
    #print(sample.shape)
    #print(base_mean)
    #print(base_motion)
    motions = sample2motion(sample,repeat(base_motion),conversion_key)
    #print(len(motions))
    #assert False
    
    z = z.detach().cpu().numpy()
    inds = np.arange(n_samples)
    cs = cmap(norm(inds))

    
    mosaic = [[f"sample {ii*n_samples_dim + jj}" for ii in range(n_samples_dim)] for jj in range(n_samples_dim)]                
    

    fig, axes = plt.subplot_mosaic(mosaic,figsize=(20,20),sharex=True,sharey=True,gridspec_kw={'wspace':0.01,'hspace':0.01},subplot_kw=dict(projection='3d'))

    for ii in range(n_samples):
        ax = axes[f"sample {ii}"]
        #ax.imshow(sample[ii, 0, :, :], cmap=cm,origin=origin)
        ax = plot_motion_on_ax(ax,motions[ii],joints)
        #ax.spines[['right','left','top','bottom']].set_color(cmap(norm(ii)))
        #ax.spines[['right','left','top','bottom']].set_linewidth(4)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_zticks([])


    if show:
        plt.show()
    else:
        plt.savefig(fn)
    plt.close()
		