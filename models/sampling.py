import numpy as numpy
import torch


def roberts_sequence(
        num_points,
        num_dims,
        root_iters=10_000,
    ):
    """
    Creates random numbers tiling a hybercube [0, 1]^d where d is `num_dims`.

    Code modified from:
    https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab
    """

    # Compute the unique positive root of f using the Newton-Raphson method.
    def f(x):
        return x ** (num_dims + 1) - x - 1

    def grad_f(x):
        return (num_dims + 1) * (x ** num_dims) - 1

    # Main loop.
    x = 1.0
    for i in range(root_iters):
        x = x - f(x) / grad_f(x)

    # Compute basis parameter
    basis = 1 - (1 / x ** (1 + torch.arange(0, num_dims)))

    # Return sequence without taking modulo 1
    return torch.arange(0, num_points)[:, None] * basis[None, :]

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a

def gen_fib_basis(m):
    """
    Creates random numbers tiling a cube [0,1]^2 where m is element of the fibonacci sequence
    """

    n = fib(m)
    z = torch.tensor([1.,fib(m-1)])

    return torch.arange(0,n)[:,None]*z[None,:]/n


def gen_korobov_basis(a,
                      num_dims,
                      num_points):
    """
    Creates `num_points` random numbers tiling a cube [0,1]^d where d is `num_dims`
    
    some recommended values:
    num_points = 1021, a = 76
    num_points = 2039, a = 1487
    num_points = 4093, a = 1516
    see table 16.1 of owens for more
    these were constructed for num_dims \in {8,12,24,32}
    this is a fibonacci lattice for num_dims = 2, a = Fib(m-1), n = Fib(m) for m >= 3
    """

    z = torch.tensor([a**k % num_points for k in range(num_dims)])
    base_pts = torch.arange(0,num_points)[:,None] * z[None,:]/num_points
    return base_pts

