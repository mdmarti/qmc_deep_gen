from stochman.manifold import Manifold
from stochman.geodesic import geodesic_minimizing_energy
from stochman.curves import CubicSpline
import torch
from scipy.interpolate import LSQBivariateSpline as LSQ
import numpy as np
from analysis.model_helpers import torus_forward,torus_reverse
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path,dijkstra 
from typing import Tuple


def prob_ratio(x: torch.FloatTensor,y:torch.FloatTensor,eps:float=1e-15) -> torch.FloatTensor:
    """
    Return probability ratio between two points x,y. This should be
    applied to the aggregated posterior over the lattice. This will be used as a measure of
    `cost`: it should be very costly to move from a high-probability point to a low probability
    point, and vice versa.

    x: posterior density of point you are moving from
    y: posterior density of point you are moving to
    eps: fudge factor to prevent division by zero

    returns: x/(y+eps), probability ratio between points x and y
    """

    return x/(y+eps)

class PosteriorDensityManifold(Manifold):
    """
    unused in current implementation, but uses 
    `stochman` package to solve approximate shortest paths in 
    continuous space.
    """

    def __init__(self,density):

        self.density = density
    def metric(self,c,return_deriv=False):

        N,D = c.shape
        c = c.detach().numpy()
        assert D == 2, print(c.shape)

        p = self.density(c[:,0],c[:,1])
        p = torch.from_numpy(p)
        m = p.pow(-2/D)
        G = m.view(-1,1).repeat(1,D)

        return G


def get_minimizing_curve(grid,posterior,p0,p1):
    """
    unused in current implementation, but uses 
    `stochman` package to solve approximate shortest paths in 
    continuous space. Constructs a manifold using 
    a spline approximation of the posterior distribution over `grid`.
    finds the connecting geodesic between any two points.
    """
    x,y = grid[:,0],grid[:,1]
    xknots = np.linspace(0,1,20)
    yknots = np.linspace(0,1,20)
    spline = LSQ(x,y,posterior,xknots,yknots)
    
    density = lambda x,y: spline(x,y,grid=False)

    manif = PosteriorDensityManifold(density)

    c,_ = manif.connecting_geodesic(torch.from_numpy(p0).to(torch.float32),torch.from_numpy(p1).to(torch.float32))

    t = torch.linspace(0,1,50)
    curve = c(t).detach().cpu().numpy()

    return curve

def construct_lattice_graph(lattice:np.ndarray,density:np.ndarray) -> np.ndarray:
    """
    takes in a lattice and a density over that lattice. returns a weighted graph over that lattice.
    This graph is constructed by embedding the lattice in the torus (sin, cos) basis, then connecting 
    the 8 nearest neighbors (in the future, this should be more precisely the points at the center
    of the voronoi cells around the center point) based on euclidean distance. Weights are then calculated by
    probability ratio of neighboring points.

    lattice: lattice over the latent space
    density: probability density over the lattice

    returns weighted_graph: connectivity graph over the lattice,
    of each points 8 NNs weighted by probability ratio.
    """


    points = torus_forward(lattice)
    nn = NearestNeighbors(n_neighbors=8,n_jobs=16)
    nn.fit(points)

    neighbor_graph = np.zeros((len(points),len(points)))
    inds = nn.kneighbors(return_distance=False)
    for row,ind in enumerate(inds):
        neighbor_graph[row,ind] = 1

    weights = []
    for p in density:
        weights.append(prob_ratio(p,density)[:,None])
    weights = np.hstack(weights)

    weighted_graph = weights * neighbor_graph

    return weighted_graph


def run_dijkstra(lattice:np.ndarray,node1_ind:int,node2_ind:int,graph:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """
    runs Dijkstra's algorithm to find shortest paths over a weighted graph. finds the shortest path between
    node_1, node_2 (as specified by their lattice indices). Returns the distance of all points from node_2,
    and the path across the lattice from node_1 to node_2

    lattice: lattice over the latent space
    node1_ind: lattice index of node_1
    node2_ind: lattice index of node_2
    graph: weighted graph over the lattice, constructed by `construct_lattice_graph`

    returns 
    dists: array of distances of all points from node_2
    path: array of all lattice points visited on the shortest path between two points
    """

    dists,predecessors = shortest_path(graph,directed=True,
                                       indices=node1_ind,return_predecessors=True)
    path = []
    ii = node2_ind
    while ii != node1_ind:
        path.append(lattice[ii])
        ii = predecessors[ii]
    path.append(lattice[node1_ind])

    return dists[node2_ind],np.vstack(path)

def get_linear_path(point_1:np.ndarray,point_2:np.ndarray,path_len:int) -> np.ndarray:
    """
    returns the linear path between two points, with `path_len` points. wraps around the edge of the latent space.

    point_1: path starting point
    point_2: path endpoint
    path_len: number of points on the path

    returns the linear path between these points (over our periodic latent space)
    """

    d = point_1.shape[-1]
    path = []
    #print(point_1,point_2)
    for ii in range(d):

        p1,p2 = point_1[ii],point_2[ii]
        #print(p1,p2)

        if p1 - p2 < -0.5:
            path.append(np.linspace(p1+1,p2,path_len)%1)
        elif p1 - p2 > 0.5:
            path.append(np.linspace(p1,p2+1,path_len)%1)
        else:
            path.append(np.linspace(p1,p2,path_len))
    #print(path)
    #path.reverse()
    #print(path)
    return np.stack(path,axis=-1)

