from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from sklearn.cluster._mean_shift import *
import numpy as np
#from sklearn.utils.validation import check_is_fitted, validate_data

class WeightedMeanShift(MeanShift):
    """
    Implement weighted mean shift. Normally, you're using this algorithm
    to estimate density from samples, move modes to centers of mass 
    of your samples. Here, we have probability densities and uniform grid,
    so we want to move modes to centers of mass using those densities on the grid
    """


    def __init__(
        self,
        *,
        bandwidth=None,
        seeds=None,
        bin_seeding=False,
        min_bin_freq=1,
        cluster_all=True,
        n_jobs=None,
        max_iter=300,
        metric = 'minkowski'
    ):

        super(WeightedMeanShift,self).__init__(bandwidth=bandwidth,
                                           seeds=seeds,
                                           bin_seeding=bin_seeding,
                                           min_bin_freq=min_bin_freq,
                                           cluster_all=cluster_all,
                                           n_jobs=n_jobs,
                                           max_iter=max_iter
        
        )
        self.metric=metric

    def fit(self, X, y=None,weights=None):
        """Perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
               Fitted instance.
        """
        #X = validate_data(self, X)
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, n_jobs=self.n_jobs)

        seeds = self.seeds
        if seeds is None:
            if self.bin_seeding:
                seeds = get_bin_seeds(X, bandwidth, self.min_bin_freq)
            else:
                seeds = X
        n_samples, n_features = X.shape
        center_intensity_dict = {}

        # We use n_jobs=1 because this will be used in nested calls under
        # parallel calls to _mean_shift_single_seed so there is no need for
        # for further parallelism.
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1,metric=self.metric).fit(X)

        # execute iterations on all seeds in parallel
        all_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_mean_shift_single_seed)(seed, X, nbrs, self.max_iter,weights)
            for seed in seeds
        )
        # copy results in a dictionary
        for i in range(len(seeds)):
            if all_res[i][1]:  # i.e. len(points_within) > 0
                center_intensity_dict[all_res[i][0]] = all_res[i][1]

        self.n_iter_ = max([x[2] for x in all_res])

        if not center_intensity_dict:
            # nothing near seeds
            raise ValueError(
                "No point was within bandwidth=%f of any seed. Try a different seeding"
                " strategy                              or increase the bandwidth."
                % bandwidth
            )

        # POST PROCESSING: remove near duplicate points
        # If the distance between two kernels is less than the bandwidth,
        # then we have to remove one because it is a duplicate. Remove the
        # one with fewer points.

        sorted_by_intensity = sorted(
            center_intensity_dict.items(),
            key=lambda tup: (tup[1], tup[0]),
            reverse=True,
        )
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        unique = np.ones(len(sorted_centers), dtype=bool)
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=self.n_jobs).fit(
            sorted_centers
        )
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center], return_distance=False)[
                    0
                ]
                unique[neighbor_idxs] = 0
                unique[i] = 1  # leave the current point as unique
        cluster_centers = sorted_centers[unique]

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs).fit(cluster_centers)
        labels = np.zeros(n_samples, dtype=int)
        distances, idxs = nbrs.kneighbors(X)
        if self.cluster_all:
            labels = idxs.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= bandwidth
            labels[bool_selector] = idxs.flatten()[bool_selector]

        self.cluster_centers_, self.labels_ = cluster_centers, labels
        return self

def _mean_shift_single_seed(my_mean, X, nbrs, max_iter,weights=None):
    # For each seed, climb gradient until convergence or max_iter
    if weights is None:
        weights = np.ones([X.shape[0],1])
    bandwidth = nbrs.get_params()["radius"]
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    completed_iterations = 0
    while True:
        # Find mean of points within bandwidth
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[0]
        points_within = X[i_nbrs]
        weights_within = weights[i_nbrs]
        weights_within /= np.sum(weights_within)
        if len(points_within) == 0:
            break  # Depending on seeding strategy this condition may occur
        my_old_mean = my_mean  # save the old mean
        my_mean = np.sum(weights_within * points_within,axis=0) #np.mean(points_within*weights_within, axis=0)
        # If converged or at max_iter, adds the cluster
        if (
            np.linalg.norm(my_mean - my_old_mean) <= stop_thresh
            or completed_iterations == max_iter
        ):
            break
        completed_iterations += 1
    return tuple(my_mean), len(points_within), completed_iterations

   