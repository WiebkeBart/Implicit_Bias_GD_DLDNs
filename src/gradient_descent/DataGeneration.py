"""Module provides the data generating function for our experiments on DLNs"""
import numpy as np
import random

def vector(N, M, s, positive=True, normalized=True):
    """function generates random data for DLNs of the form y=Ax, with x\in R^N s-sparse (Gaussian), y\in R^M, A (Gaussian)

    Parameters:
        N (int)                 - dimension of the parameters (x)
        M (int)                 - number of data points (size of y) should be M<<N
        s (int)                 - sparsity level of x, should be (1, ..., N)
        positive (boolean)      - whether x is enforced to have positive entries
        normalized (boolean)    - whether A and x are normalized
    Returns:
        A,x,y                   - with Ax = y (random Gaussian Data)
    """

    A = np.random.normal(size=(M, N))
    if normalized:
        A /= np.linalg.norm(A, axis=0)
    #else:
    #    A /= np.sqrt(M)
    x = np.random.normal(size=N)
    x[s:] = 0
    if positive:
        x = np.abs(x)
    random.shuffle(x)
    if normalized:
        x = x / np.linalg.norm(x)
    y = np.matmul(A, x)
    return A, x, y