import third_party.amc_parser as amc
import numpy as np 

def getRotation(joints,motion,excluded = ['toes','hand','fingers','thumb','hipjoint']):
	
	rotations = []
	globalrots = []
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
				#print(motion['root'])
				#print(len(motion['root']))
				globalRot = np.deg2rad(motion['root'][3:])
				#print(len(rotation))
				globalrots.append(globalRot)
			else:
				idx = 0
				rotation = []
				for axis, lm in enumerate(joint.limits):
					if not np.array_equal(lm, np.zeros(2)):
						rotation.append(motion[joint.name][idx])
						idx += 1
				#print(joint.name)
				rotation = np.hstack(rotation)
				rotation = np.deg2rad(rotation)

				rotations.append(rotation)
	#rotations.append(globalRot)
	return np.hstack(rotations),np.array(globalrots)

def preprocess_mocap(jointList,motionsList,n_frames_per_sample=3):
	
	"""
	this preprocessing follows the steps from "Gaussian Process Dynamical Models for Human Motion" (Wang, Fleet, & Hertzmann 2008)
	Briefly, we treat each pose a 44 Euler angles for joints, three global (torso) pose angles,
	and 3 global (torso) translational velocities.
	All data are then mean-subtracted.
	we treat each data point as a set of n frames over time, so we stack after mean subtracting
	"""

	allTrajs = []
	for subject,motions in zip(jointList,motionsList):

		subjTrajs = []
		for trial in motions:
			rotTrial,globalRotTrial = [],[]
			for frame in trial:
				rot,globalrot = getRotation(subject,frame)
				rotTrial.append(rot)
				globalRotTrial.append(globalrot)

			rotTrial = np.vstack(rotTrial)
			globalRots = np.vstack(globalRotTrial)
			globalTranslation = np.diff(globalRots,axis=0)
			globalTranslation = np.vstack([globalTranslation,globalTranslation[-1:,:]])
			allRots = np.hstack([rotTrial,globalRots,globalTranslation])
			allRots = allRots - np.nanmean(allRots,axis=0)

			trial_converted = frames2samples(allRots,n_frames_per_sample)
			subjTrajs.append(trial_converted)

		allTrajs = allTrajs + subjTrajs

	return allTrajs

def frames2samples(frames,n_frames_per_sample):

	n_frames = len(frames)
	samples = []
	for start in range(0,n_frames - n_frames_per_sample):

		samples.append(frames[start:start+n_frames_per_sample])

	return np.stack(samples,axis=0)