from analysis.weighted_mean_shift import *
from sklearn.cluster import MeanShift,KMeans  
import os
from analysis.model_helpers import torus_reverse

import numpy as np
#from sklearn.utils.validation import check_is_fitted, validate_data
def p_dist_theta(x,y,p=2):
    """
    only for thetas bounded between 0,1
    """

    d1 = np.abs(x-y)
    d2 = np.abs(x - (1+y))
    d3 = np.abs((1+x) - y)
    abs_dist = np.minimum(np.minimum(d1,d2),d3)
    return (abs_dist ** p).sum() ** (1/p)

def p_dist_theta_alternate(x,y,p=2):

    d1 = np.abs(x-y)
    b1 = d1 > 0.5
    b2 = ((x > 0.5) -0.5)*2

    abs_dist = np.abs(x-y - b1*b2)
    return (abs_dist**p).sum()**(1/p)

def run_mean_shift(latent_points,seeds,weights,bandwidth,n_jobs,p,embedded=True,normal=False):

    #print(np.amax(seeds),np.amin(seeds))
    #print(np.amax(latent_points),np.amin(latent_points))
    if embedded:
        metric = 'minkowski'
        print('using normal wms')
        wms = WeightedMeanShift(n_jobs=n_jobs,bandwidth=2*bandwidth,metric=metric,seeds=seeds)
    else:
        metric = lambda x,y: p_dist_theta(x,y,p=p)
        print('using circular wms')
        wms = WeightedMeanShiftCircular(n_jobs=n_jobs,bandwidth=bandwidth,metric=metric,seeds=seeds)
    if normal:
        wms.fit(latent_points)
    else:
        wms.fit(latent_points,weights=weights,verbose=False)
    centers = wms.cluster_centers_
    if embedded:
        #print(np.amax(centers),np.amin(centers))
        
        centers = torus_reverse(centers)
        #print(np.amax(centers),np.amin(centers))
        #print(centers.shape)

    return centers,wms

def run_kmeans(embeddings,n_clusters,norm_order=2):

    km = KMeans(n_clusters=n_clusters)