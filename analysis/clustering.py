from analysis.weighted_mean_shift import *
from sklearn.cluster import MeanShift,KMeans  
import os

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

def run_mean_shift(latent_points,seeds,weights,bandwidth,n_jobs,p):

    metric = lambda x,y: p_dist_theta(x,y,p=p)
    wms = WeightedMeanShift(n_jobs=n_jobs,metric=metric,bandwidth=bandwidth,seeds=seeds)
    wms.fit(latent_points,weights=weights)
    centers = wms.cluster_centers_

    return centers,wms

def run_kmeans(embeddings,n_clusters,norm_order=2):

    km = KMeans(n_clusters=n_clusters)