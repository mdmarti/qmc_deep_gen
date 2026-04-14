from analysis.weighted_mean_shift import *
from sklearn.cluster import KMeans  
from typing import Union,Tuple
from analysis.model_helpers import torus_reverse

import numpy as np
#from sklearn.utils.validation import check_is_fitted, validate_data
def p_dist_theta(x:np.ndarray,y:np.ndarray,p:int=2) -> float:
    """
    only for thetas bounded between 0,1. computes d-dimensional p-norm 
    on a (0,1) bounded periodic domain (our latent space)
    x: point one
    y: point 2
    p: order of p-norm

    returns: p-norm of difference between x,y
    """

    d1 = np.abs(x-y)
    d2 = np.abs(x - (1+y))
    d3 = np.abs((1+x) - y)
    abs_dist = np.minimum(np.minimum(d1,d2),d3)
    return (abs_dist ** p).sum() ** (1/p)

def p_dist_theta_alternate(x,y,p=2):
    """
    only for thetas bounded between 0,1. computes d-dimensional p-norm 
    on a (0,1) bounded periodic domain (our latent space)
    x: point one
    y: point 2
    p: order of p-norm

    returns: p-norm of difference between x,y. alternate version, works just about the same
    as the other version
    """

    d1 = np.abs(x-y)
    b1 = d1 > 0.5
    b2 = ((x > 0.5) -0.5)*2

    abs_dist = np.abs(x-y - b1*b2)
    return (abs_dist**p).sum()**(1/p)

def run_mean_shift(latent_points:np.ndarray,seeds:np.ndarray,
                   weights:Union[np.ndarray,list],bandwidth:float,
                   n_jobs:int=1,p:int=2,
                   embedded:bool=True,weighted:bool=True) -> Tuple[np.ndarray,Union[WeightedMeanShift,WeightedMeanShiftCircular],Union[list,np.ndarray]]:
    """
    runs (weighted) mean shift on latent_points. this method clusters latent points
    based on their density, automatically selecting a number of clusters. The number of clusters
    found depends on `bandwidth` (this determines where clusters move and when to merge them) and
    `seeds`, where the number of clusters found is upper bounded by the number of seeds. By default,
    performs weighted mean shift in the basis-embedded space, so that we can use euclidean distance
    and leverage the nice fast performance of norms in sklearn

    latent_points: points in the latent space. if using weighted mean shift, these should be lattice points.
    if using normal mean shift, these should be data representations in the latent space.
    seeds: cluster initializations. You should start with a good number of these, more clusters than you expect there
    to be, then let the algorithm whittle them down
    weights: density weights on the lattice. this should only be used in weighted mean-shift, if using normal mean shift just pass
    an empty list
    bandwidth: hyperparameter for mean-shift algorithm. Sets the neighborhood over which density is calculated, and within 
    which different clusters are merged. You should run this function for multiple values of bandwidth
    n_jobs: number of jobs for parallelization. If you want to parallelize, increase this from 1
    p: p-norm to calculate distances. 2 is euclidean, I haven't tried others
    embedded: whether to run mean-shift in basis-embedded space or latent space (periodic)
    weighted: whether to run weighted or normal mean-shift

    TO DO: fix labels! I don't think prediction is working right now....

    returns:
    centers: the centroids found by mean-shift
    wms: (weighted) mean shift object
    labels: nothing right now, will eventually be the flowed cluster assignments of all latent points.
    """

    if embedded:
        metric = 'minkowski'
        print('using L_p metrics')
        wms = WeightedMeanShift(n_jobs=n_jobs,bandwidth=2*bandwidth,metric=metric,seeds=seeds)
    else:
        metric = lambda x,y: p_dist_theta(x,y,p=p)
        print('using circular metric')
        wms = WeightedMeanShiftCircular(n_jobs=n_jobs,bandwidth=bandwidth,metric=metric,seeds=seeds)
    #no labels for now.... fix later
    if not weighted:
        print("fitting regular mean shift")
        wms.fit(latent_points)
        #labels = wms.predict(latent_points)
    else:
        print('fitting weighted mean shift')
        wms.fit(latent_points,weights=weights,verbose=False)
        #labels = wms.predict(latent_points,weights=weights,verbose=False,max_iter=300,tol=1e-2)
    centers = wms.cluster_centers_
    if embedded:
        
        centers = torus_reverse(centers)

    labels = []

    return centers,wms,labels

def run_kmeans(embeddings,n_clusters,norm_order=2):

    km = KMeans(n_clusters=n_clusters)