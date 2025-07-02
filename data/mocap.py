import third_party.amc_parser as amc
import numpy as np 
import os
import glob
from tqdm import tqdm 
import copy 
import matplotlib.pyplot as plt 

def getRotation(joints,motion,excluded = ['toes','hand','fingers','thumb','hipjoint']):
    
    rotations = []
    globalrots = []
    jointKey = []
    #print(len(joints))
    for j in joints.keys():
        joint = joints[j]
        drop = False
        for kk in excluded:
            if kk in joint.name:
                drop = True
                #print(f'dropping {joint.name}')
                continue
        if not drop:
            if joint.name == 'root':

                globalRot = np.deg2rad(motion['root'][3:])

                globalrots.append(globalRot)
                jointKey.append((j,len(globalRot)))
            else:
                idx = 0
                rotation = []
                for axis, lm in enumerate(joint.limits):
                    if not np.array_equal(lm, np.zeros(2)):
                        rotation.append(motion[joint.name][idx])
                        idx += 1
                #print(joint.name)
                rotation = np.hstack(rotation)
                #print(f"original values {joint.name}: {rotation}")

                rotation = np.deg2rad(rotation)

                rotations.append(rotation)
                jointKey.append((j,len(rotation)))

    return np.hstack(rotations),np.array(globalrots),jointKey

def preprocess_mocap_motion(joints,motion,n_frames_per_sample=3):

    ### only process one motion at a time. this is getting too complicated
    """
    this preprocessing follows the steps from "Gaussian Process Dynamical Models for Human Motion" (Wang, Fleet, & Hertzmann 2008)
    Briefly, we treat each pose a 44 Euler angles for joints, three global (torso) pose angles,
    and 3 global (torso) translational velocities.
    All data are then mean-subtracted.
    we treat each data point as a set of n frames over time, so we stack after mean subtracting
    """

    

    rotTrial,globalRotTrial = [],[]
    for frame in motion:
        #print(subject,frame)
        rot,globalrot,jointNames = getRotation(joints,frame)
        
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
    allRots = allRots - muRot
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

def get_samples(datapath,subject,n_frames_per_sample=4):
	
	
	asf = glob.glob(os.path.join(datapath,subject,'*.asf'))[0]
	amcs = glob.glob(os.path.join(datapath,subject,'*.amc'))
	file_numbers = [int(a.split('_')[-1].split('.amc')[0]) for a in amcs]
	order = np.argsort(file_numbers)
	amcs = [amcs[o] for o in order]
	
	frames = [amc.parse_amc(am) for am in amcs]
	joints = amc.parse_asf(asf) # for asf in asfs]
	
	trials,means,keys,frame_nos,labels = [],[], [],[],[]
	
	for ii,trial in tqdm(enumerate(frames,total=len(frames))):
		traj,mean,key = preprocess_mocap_motion(joints,trial,n_frames_per_sample=n_frames_per_sample)
	
		trials.append(traj)
		means.append(mean)
		keys.append(key)
		frame_nos.append(np.arange(len(traj)))
		labels.append(ii * np.ones((len(traj),)))
		
	return trials,means,keys,frames,joints,frame_nos,labels

		
		
def sample2motion(sample,motion,conversion_key,testing = False):

	"""
	converts a trajectory back into a motion, for plotting purposes.
	will replace the values for all joints with the model predictions.
	This assumes that trajectories have already been re-meaned.
	
	this also only takes one sample, one motion at a time -- and assumes the samples in the motion correspond to the samples in the sample!!

	something is going wrong here. not sure what. Return the un-mean-centered data in addition to the mean-centered so we can see what's up
	"""
	"""
	remove the assertions for when 
	"""
	#print(
	traj = np.rad2deg(sample)
	newMotion = [] #copy.deepcopy(motion)
	
	for sampleFrame,motionFrame in zip(traj,motion):

		newFrame = copy.deepcopy(motionFrame)
		start_ind = 0
		for (jointName,DoF) in conversion_key:
			# these are added to the list in the same order that joints are added to the sample list, so this should be okay
			#print(f"replacing {jointName}")
			if jointName == 'root':
				newFrame[jointName] = motionFrame[jointName][:3] + sampleFrame[start_ind:start_ind + DoF].tolist() # unsure if this will work, but it should
				if testing:
					assert np.all(np.isclose(np.array(motionFrame[jointName][3:]),np.array(sampleFrame[start_ind:start_ind + DoF].tolist()))),print(motionFrame[jointName][3:],sampleFrame[start_ind:start_ind + DoF].tolist(),start_ind,start_ind+DoF)
			else:
				newFrame[jointName] = sampleFrame[start_ind:start_ind + DoF].tolist()
				if testing:
					assert np.all(np.isclose(sampleFrame[start_ind:start_ind + DoF],np.array(motionFrame[jointName]))),print(sampleFrame[start_ind:start_ind + DoF],np.array(motionFrame[jointName]),start_ind,start_ind+DoF)

			start_ind += DoF
		newMotion.append(newFrame)

	return newMotion

def plot_sample(sample,joints,motion,conversion_key,n_per_sample=5):

	
	predicted_motion = sample2motion(sample,motion,conversion_key)
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

	plt.show()
	plt.close()
		