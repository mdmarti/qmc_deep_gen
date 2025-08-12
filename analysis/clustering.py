from weighted_mean_shift import *
from sklearn.cluster import MeanShift,KMeans  
import os

import numpy as np
#from sklearn.utils.validation import check_is_fitted, validate_data
def p_norm_theta(x,y,p=2):
    """
    only for thetas bounded between 0,1
    """

    d1 = np.abs(x-y)
    d2 = np.abs(x - (1+y))
    abs_dist = np.minimum(d1,d2)
    return (abs_dist ** p).sum() ** (1/p)


def run_mean_shift(embeddings,seeds,norm_order=2):

    n_workers = len(os.sched_getaffinity(0))
    metric = lambda x,y: p_norm_theta(x,y,p=norm_order)
    ms = MeanShift(seeds=seeds,n_jobs = n_workers,metric=metric)
    ms.fit(embeddings)

    return ms.cluster_centers_,ms.labels_


def run_kmeans(embeddings,n_clusters,norm_order=2):

    km = KMeans(n_clusters=n_clusters)