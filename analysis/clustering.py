from weighted_mean_shift import *


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


def run_mean_shift(embeddings,norm_order=2):

    pass

def run_kmeans(embeddings,norm_order=2):

    pass