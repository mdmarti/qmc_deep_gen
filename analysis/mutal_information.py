from sklearn.neighbors import KernelDensity
import numpy as np


def estimate_density(data,bandwidth,kernel):

    kde = KernelDensity(bandwidth=bandwidth,kernel=kernel)

    kde.fit(data)

    log_density = kde.score(data)

    return log_density

def mutual_info_kde(latent_embeddings,values):


    lp_x1 = estimate_density(latent_embeddings)
    lp_x2 = estimate_density(values)

    lp_x1x2 = estimate_density(np.hstack([latent_embeddings,values]))

    ratio = lp_x1x2 - lp_x1 - lp_x2 

    mi = np.sum(np.exp(lp_x1x2) * ratio)

    return mi
