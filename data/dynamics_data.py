from abc import ABC,abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.interpolate import interp2d
import warnings
from scipy.io.wavfile import WavFileWarning
import sys
sys.path.append('/hdd/miles/AMCParser')
import amc_parser as amc
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter

EPSILON = 1e-12

def z_score(data):

	x_stacked =np.vstack(data)
	mu = np.nanmean(x_stacked,axis=0,keepdims=True)
	sd = np.nanstd(x_stacked,axis=0,keepdims=True)

	return [(d - mu)/sd for d in data],mu,sd

def scale(data):
	x_stacked =np.vstack(data)
	sd = np.nanstd(x_stacked,axis=0,keepdims=True)
	mag = np.amax(np.abs(x_stacked))
	return [d/mag for d in data],mag

def plot_mocap_gif(joint,motion,latents=[],fn='testani.mp4'):

	#fig = plt.figure(figsize=(15,5))
	
	if len(latents) == 0:
		fig,jointAx = plt.subplots(nrows=1,ncols=1,subplot_kw=dict(projection="3d"),layout='compressed')
		#jointAx = fig.add_subplot(111,projection='3d')
		
		def animate(i,scatterAndLines,motion,joint):

			#print(scatterAndLines)
			if i < 2*len(motion):
				joint['root'].set_motion(motion[i//3])
			pts = joint['root'].to_dict()
			xs,ys,zs = [],[],[]
			for j in pts.values():
				xs.append(j.coordinate[0,0])
				ys.append(j.coordinate[1,0])
				zs.append(j.coordinate[2,0])
				scatterAndLines[0].set_data(zs,xs)#
				scatterAndLines[0].set_3d_properties(ys)
			
			#jointAx.plot(zs,xs,ys,'b.')
			counter = 1
			for j in pts.values():
				child = j
				if child.parent is not None:
					parent = child.parent
					xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
					ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
					zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
					scatterAndLines[counter].set_data(zs,xs)
					scatterAndLines[counter].set_3d_properties(ys)
					counter += 1
			
			return scatterAndLines
	
		jointAx.set_xlim3d(-35, 55)
		jointAx.set_ylim3d(-20, 40)
		jointAx.set_zlim3d(-20, 40)
		joint['root'].set_motion(motion[0])
		pts = joint['root'].to_dict()
		xs,ys,zs = [],[],[]
		ls = []
		for j in pts.values():
			xs.append(j.coordinate[0,0])
			ys.append(j.coordinate[1,0])
			zs.append(j.coordinate[2,0])
		l = jointAx.plot(zs,xs,ys,'b.')
		print(l)
		ls.append(l[0])
		jointAx.view_init(azim=90,elev=10)
		for iter,j in enumerate(pts.values()):
			child = j
			if child.parent is not None:
				parent = child.parent
				xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
				ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
				zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
				l = jointAx.plot(zs,xs,ys,'-r')
				ls.append(l[0])
		anim = lambda i: animate(i,ls,motion,joint)
		
	else:
		fig,(jointAx,latAx) = plt.subplots(nrows=1,ncols=2,subplot_kw=dict(projection="3d"),layout='compressed')
		#jointAx = fig.add_subplot(121,projection='3d')
		#latAx = fig.add_subplot(122,projection='3d')

		def animate(i,scatterAndLines,motion,joint,latents,latentAxis):

			#print(scatterAndLines)
			if i < 2*len(motion):
				joint['root'].set_motion(motion[i//2])
				d = latents[i//2,:]
			else:
				d = latents[-1,:]
			pts = joint['root'].to_dict()
			xs,ys,zs = [],[],[]
			for j in pts.values():
				xs.append(j.coordinate[0,0])
				ys.append(j.coordinate[1,0])
				zs.append(j.coordinate[2,0])
				scatterAndLines[0].set_data(zs,xs)#
				scatterAndLines[0].set_3d_properties(ys)
			
			#jointAx.plot(zs,xs,ys,'b.')
			counter = 1
			for j in pts.values():
				child = j
				if child.parent is not None:
					parent = child.parent
					xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
					ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
					zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
					scatterAndLines[counter].set_data(zs,xs)
					scatterAndLines[counter].set_3d_properties(ys)
					counter += 1
			
			(xs,ys,zs) = scatterAndLines[-1]._offsets3d
			xs,ys,zs = np.hstack([xs,[d[0]]]),np.hstack([ys,[d[1]]]),np.hstack([zs,[d[2]]])
			scatterAndLines[-1]._offsets3d = (xs,ys,zs)
			if i > 2*len(motion):
				latentAxis.view_init(azim=(i - 2*len(motion))*0.25,elev=10)

			return scatterAndLines
	
		jointAx.set_xlim3d(-35, 55)
		jointAx.set_ylim3d(-20, 40)
		jointAx.set_zlim3d(-20, 40)
		joint['root'].set_motion(motion[0])
		pts = joint['root'].to_dict()
		xs,ys,zs = [],[],[]
		ls = []
		for j in pts.values():
			xs.append(j.coordinate[0,0])
			ys.append(j.coordinate[1,0])
			zs.append(j.coordinate[2,0])
		l = jointAx.plot(zs,xs,ys,'b.')
		print(l)
		ls.append(l[0])
		jointAx.view_init(azim=90,elev=10)
		for iter,j in enumerate(pts.values()):
			child = j
			if child.parent is not None:
				parent = child.parent
				xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
				ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
				zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
				l = jointAx.plot(zs,xs,ys,'-r')
				ls.append(l[0])
		latAx.set_xlim3d(np.amin(latents[:,0] - 1), np.amax(latents[:,0]+1))
		latAx.set_ylim3d(np.amin(latents[:,1] - 1), np.amax(latents[:,1]+1))
		latAx.set_zlim3d(np.amin(latents[:,2] - 1), np.amax(latents[:,2]+1))
		l = latAx.scatter(latents[0,0],latents[0,1],latents[0,2])
		latAx.view_init(azim=0,elev=10)
		ls.append(l)

		anim = lambda i: animate(i,ls,motion,joint,latents,latAx)


	#fig.tight_layout()
	ani = animation.FuncAnimation(fig,anim,frames=len(motion)*3,interval=50,blit=True)
	Writer=animation.writers['ffmpeg']
	writer = Writer(fps=30,bitrate=500)
	ani.save(fn,writer=writer,dpi=400)
	plt.close(fig)
	return

	

	

	

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

def preprocess_mocap(jointList,motionsList):

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

			subjTrajs.append(allRots)

		allTrajs = allTrajs + subjTrajs

	return allTrajs



class ToyData(ABC):

    """
    generic base class for toy datasets. implements
    generate, which all classes need, and defines methods needed
    by all inheriting classes
    """


    def __init__(self):

        pass

    @abstractmethod
    def f(self,x,t):
        raise NotImplementedError

    @abstractmethod
    def g(self,x,t):
        raise NotImplementedError

    @abstractmethod
    def dW(self,dt):
        raise NotImplementedError 

    @abstractmethod
    def init_conditions(self):
        raise NotImplementedError

    def dx(self,x,t,dt,sigma):

        fx = self.f(x,t)
        gx = self.g(x,t,sigma)
        dw = self.dW(dt)

        return fx * dt + gx @ dw
    
    def generate(self,n,T,dt,sigma):

        trajectories = []
        t = np.arange(dt,T+dt/2,dt)
        for ii in range(n):

            xnot = self.init_conditions()
            x = [xnot]

            for jj in range(1,len(t)):

                xx = x[jj-1]
                tt = t[jj-1]

                dx = self.dx(xx,tt,dt,sigma)
                
                x.append(xx + dx)
            
            x = np.vstack(x)
            trajectories.append(x)

        return trajectories
    
class Vanderpol(ToyData):

    def __init__(self,coeffs=[2,15],seed=1234):

        super(Vanderpol,self).__init__()
        self.rho,self.tau = coeffs
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        dx1 = self.rho * self.tau * (x[0] - x[0]**3/3 - x[1])
        dx2 = self.tau/self.rho * x[0]
        return np.hstack([dx1,dx2])
    def g(self,x,t,sigma):
        return sigma*np.eye(2)
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((2,)),cov=dt*np.eye(2))

    def init_conditions(self):

        return self.gen.multivariate_normal(mean=[1,1],cov=np.eye(2)*0.03)
    
class DoubleCircles(ToyData):

    def __init__(self,coeffs=[3.5,4,2*np.pi],seed=1234):

        super(DoubleCircles,self).__init__()

        self.r0,self.a,self.omega = coeffs
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        print("f unused in double circles")
        pass

    def ft(self,theta,r,t):
        #dtheta = omega
        if r > self.r0:
            return self.omega
        else:
            return -self.omega
    
    def fr(self,r,t):
        
        # potential function: (r - r0)^4 - a(r - r0)^2
        return -(4 * (r - self.r0)**3 - 2*self.a * (r - self.r0))

    def g(self,x,t,sigma):
        return sigma*np.eye(2)

    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((2,)),cov = dt*np.eye(2))
    
    def _polar_to_cartesian(self,r,theta):
    
        return np.hstack([r*np.cos(theta),r*np.sin(theta)])
    
    def _cartesian_to_polar(self,xy):
        
        r = np.linalg.norm(xy)
        theta = np.arctan2(xy[1],xy[0])
        return r,theta
    
    def init_conditions(self):
        r0 = self.r0 + self.gen.normal(loc=0,scale=0.1)
        t0 = self.gen.uniform(0,2*np.pi)

        return self._polar_to_cartesian(r0,t0)

    def dx(self,x,t,dt,sigma):
        
        r,theta = self._cartesian_to_polar(x)
        dr = self.fr(r,t)
        dtheta = self.ft(theta,r,t)
        r += dr*dt 
        theta += dtheta*dt

        newX = self._polar_to_cartesian(r,theta)
        dw_xy = self.g(x,t,sigma) @ self.dW(dt)
        xy2 = newX + dw_xy
        
        return xy2 - x

class Lorenz63(ToyData):

    def __init__(self,coeffs=[10,28,8/3],seed=1234):

        super(Lorenz63,self).__init__()
        self.sigma,self.rho,self.beta = coeffs
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        
        dx = self.sigma * (x[1] - x[0]) #+ sample_dW[0]
        dy = (x[0] * (self.rho - x[2]) - x[1]) #+ sample_dW[1]
        dz = (x[0]*x[1]  - self.beta*x[2]) #+ sample_dW[2]
        return np.hstack([dx,dy,dz])
    
    def g(self,x,t,sigma):
        return sigma*np.eye(3)
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((3,)),cov=dt*np.eye(3))
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=np.zeros((3,)),cov=np.eye(3))

class Lorenz96(ToyData):

    def __init__(self,coeffs=[8],d=10,seed=1234):

        super(Lorenz96,self).__init__()
        self.F = coeffs
        self.d = d
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        dx = np.zeros(self.d)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(self.d):
            dx[i] = (x[(i + 1) % self.d] - x[i - 2]) * x[i - 1] - x[i] + self.F
        return dx
    
    def g(self,x,t,sigma):
        return np.eye(self.d)*sigma
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((self.d,)),cov=np.eye(self.d)*dt)
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=np.zeros((self.d,)),cov=np.eye(self.d))
    
class Rossler(ToyData):

    def __init__(self,coeffs=[0.1,0.1,14],seed=1234):

        super(Rossler,self).__init__()
        self.a,self.b,self.c = coeffs
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        dx = -x[1] - x[2] #+ sample_dW[0]
        dy = x[0] + self.a*x[1] #+ sample_dW[1]
        dz = self.b + x[2]*(x[0] - self.c) #+ sample_dW[2]
        return np.hstack([dx,dy,dz])
    
    def g(self,x,t,sigma):
        return np.eye(3)*sigma
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((3,)),cov=np.eye(3)*dt)
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=[0,-9,0],cov=6*np.eye(3))

class DoubleSDE(ToyData):

    def __init__(self,coeffs=[np.pi,-np.pi,np.array([-0.5,0]),np.array([0.5,0])],seed=1234):

        super(DoubleSDE,self).__init__()
        self.omega1,self.omega2,self.center1,self.center2 = coeffs
        #self.center2 = -self.center1
        
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):
        print("f unused in diverging sdes")
        pass
    def ft(self,theta,x,t):
        if x[0] < 0:
            return self.omega1
        else:
            return self.omega2
        
    def g(self,x,t,sigma):
        return sigma * np.eye(2)#/(np.abs(x[0])+1/10)
    
    def dW(self,dt):
        return self.gen.multivariate_normal(mean=np.zeros((2,)),cov=np.eye(2)*dt)
    
    def _polar_to_cartesian(self,r,theta):
    
        return np.hstack([r*np.cos(theta),r*np.sin(theta)])
    
    def _cartesian_to_polar(self,xy):
    
        r = np.linalg.norm(xy)
        theta = np.arctan2(xy[1],xy[0])
        return r,theta
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=[0,-0.5],cov=np.eye(2)*0.01)

    def dx(self,x,t,dt,sigma):

        if x[0] < 0:
            v = x - self.center1
            r,theta = self._cartesian_to_polar(v)
            dtheta = self.ft(theta,x,t)
            theta += dtheta*dt
            xx2 = self._polar_to_cartesian(r,theta) + self.center1
        else:
            v = x - self.center2
            r,theta = self._cartesian_to_polar(v)
            dtheta = self.ft(theta,x,t)
            theta += dtheta*dt
            xx2 = self._polar_to_cartesian(r,theta) + self.center2

        return xx2 + self.g(x,t,sigma)@self.dW(dt) - x

class Balls(ToyData):

    def __init__(self,coeffs=np.array([[-1,-3],[-3,1]]),seed=1234):

        super(Balls,self).__init__()
        self.coeffs = np.array(coeffs)

        #self.center2 = -self.center1
        
        self.gen = np.random.default_rng(seed=seed)

    def f(self,x,t):

        return self.coeffs @ x 

    def g(self,x,t,sigma):

        return sigma*np.eye(2)

    def dW(self,dt):

        return self.gen.multivariate_normal(mean=np.zeros((2,)),cov=dt*np.eye(2))
    
    def init_conditions(self):

        return self.gen.multivariate_normal(mean=[-1,0],cov=np.eye(2)*0.25**2)
    
    def traj_to_movie(self,trajectories,image_shape,radius,blur=False):
        """
        converts a latent trajectory to a movie
        """


        movies = []
        gX,gY = np.meshgrid(np.linspace(-1,1,2*radius),np.linspace(-1,1,2*radius))
        ball = (gX**2 + gY**2 < 1)
        image_shape = np.array(image_shape)
        
        for traj in trajectories:
            traj -= np.amin(traj)
            traj /= np.amax(np.abs(traj))

            frames = np.zeros((len(traj),1,image_shape[0],image_shape[1]))
        
            for point in range(len(traj)):
        
                ballCenter = (traj[point,:] *(image_shape - 2*radius)).astype(int) + radius
                frames[point,0,ballCenter[0]-radius:ballCenter[0] + radius, ballCenter[1]-radius:ballCenter[1] + radius] += ball
                if blur:
                    frames[point,0,:,:] = gaussian_filter(frames[point,0,:,:],sigma=radius*4,truncate=0.05)

            movies.append(frames)

        return movies


def downsample(data:list,origdt:float,newdt:float,noise:bool=True) -> np.ndarray:

	skip = int(newdt/origdt)

	downsampled = [d[::skip] for d in data]

	if noise:
		downsampled = [d + 0.01*np.random.randn(*d.shape) for d in downsampled]

	return downsampled

class toyDataset(Dataset):

	def __init__(self,data,dt,nForward=1) -> None:
		"""
		toyData: list of numpy arrays
		"""

		self.maxForward = nForward
		exampleInd = np.random.choice(len(data),1)[0]
		self.exampleTraj = data[exampleInd]
		lens = list(map(len,data))
		lens2 = [0] + list(np.cumsum([l for l in lens][:-1]))
		sets = [np.vstack([np.arange(ii, l+ ii - self.maxForward) for ii in range(self.maxForward + 1)]).T for l in lens]
		#pairs = [np.vstack([np.arange(0,l-1),np.arange(1,l)]).T for l in lens]
		sumSets = [p+l for p,l in zip(sets,lens2)]
		validInds = np.vstack(sumSets)
		self.data= np.vstack(data)
		self.data_inds = validInds
		self.dt = dt
		self.length = len(validInds)
		#print('added in more forward predictions')
		## needed: slice data by dt? need true dt, ds dt for that
		## should be fine to add though

	def __len__(self):

		return self.length 
	
	def __getitem__(self, index):
		
		single_index = False
		result = []
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True

		for ii in index:
			inds = self.data_inds[ii]

			samples = [self.transform(self.data[ind]) for ind in inds]
			samples.append(self.dt)			
			#s1,s2 = self.transform(self.data[inds[0]]),self.transform(self.data[inds[1]])
			result.append(samples)

		if single_index:
			return result[0]
		return result
	
	def transform(self,data):
		return torch.from_numpy(data).type(torch.FloatTensor)

class toyDatasetLinearity(Dataset):

	def __init__(self,data,dt) -> None:
		"""
		toyData: list of numpy arrays
		"""
		print("Don't use this, the normal dataset now does everything this one does")

		exampleInd = np.random.choice(len(data),1)[0]
		self.exampleTraj = data[exampleInd]
		lens = list(map(len,data))
		lens2 = [0] + list(np.cumsum([l for l in lens][:-1]))
		triplets = [np.vstack([np.arange(0,l-2),np.arange(1,l-1),np.arange(2,l)]).T for l in lens]
		sumTriplets = [p+l for p,l in zip(triplets,lens2)]
		validInds = np.vstack(sumTriplets)
		self.data= np.vstack(data)
		self.data_inds = validInds
		self.dt = dt
		self.length = len(validInds)
		print('we no longer sampling now')
		## needed: slice data by dt? need true dt, ds dt for that
		## should be fine to add though

	def __len__(self):

		return self.length 
	
	def __getitem__(self, index):
		
		single_index = False
		result = []
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True

		for ii in index:
			inds = self.data_inds[ii]
						
			s1,s2,s3 = self.transform(self.data[inds[0]]),self.transform(self.data[inds[1]]),self.transform(self.data[inds[2]])
			result.append((s1,s2,s3,self.dt))

		if single_index:
			return result[0]
		return result
	
	def transform(self,data):
		return torch.from_numpy(data).type(torch.FloatTensor)
		

def makeToyDataloaders(ds1,ds2,dt,batch_size=512,t='regular',nForward=1):

	#assert ds1.shape[1] == 3
	#ds1 = ds1).type(torch.FloatTensor)
	#ds2 = torch.from_numpy(ds2).type(torch.FloatTensor)
	adjustedBatch = batch_size // nForward
	print(f"this DL will return {adjustedBatch} trajectories per batch")
	if t== 'regular':
		dataset1 = toyDataset(ds1,dt,nForward=nForward)
		dataset2 = toyDataset(ds2,dt,nForward=nForward)
	elif t =='linearity':
		print('just use the regular version dingus')
		dataset1 = toyDatasetLinearity(ds1,dt)
		dataset2 = toyDatasetLinearity(ds2,dt)
	else:
		print("What are you doing")
		return NotImplementedError


	trainDataLoader = DataLoader(dataset1,batch_size=adjustedBatch,shuffle=True,
				  num_workers=4)
	testDataLoader = DataLoader(dataset2,batch_size=adjustedBatch,shuffle=False,
				  num_workers=4)
	
	return {'train':trainDataLoader,'test':testDataLoader}
