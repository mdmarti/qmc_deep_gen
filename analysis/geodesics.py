from stochman.manifold import Manifold
from stochman.geodesic import geodesic_minimizing_energy
from stochman.curves import CubicSpline
import torch
from scipy.interpolate import LSQBivariateSpline as LSQ
import numpy as np
from analysis.model_helpers import torus_forward,torus_reverse
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path,dijkstra 


def prob_ratio(x,y,eps=1e-15):

    return x/(y+eps)

class PosteriorDensityManifold(Manifold):

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
    this will only work in 2d
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

def construct_lattice_graph(lattice,density):


    points = torus_forward(lattice)
    nn = NearestNeighbors(n_neighbors=25,n_jobs=16)
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


def run_dijkstra(lattice,node1_ind,node2_ind,graph):

    dists,predecessors = shortest_path(graph,directed=True,
                                       indices=node1_ind,predecessors=True)
    path = []
    ii = node2_ind
    while ii != node1_ind:
        path.append(lattice[ii])
        ii = predecessors[ii]
    path.append(lattice[node1_ind])

    return dists[node2_ind],np.vstack(path)




