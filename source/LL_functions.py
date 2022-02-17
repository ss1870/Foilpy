import numpy as np
import math
# from numba import jit


# @jit(nopython=True)
def eval_biot_savart(xcp, xnode1, xnode2, gamma, l0):

    xcp = xcp.reshape(-1, 1, 3)  # xcp shape (ncp, 1, 3)
    r1 = xcp - xnode1  # r1 shape (ncp, nvl, 3)
    r2 = xcp - xnode2  # r2 shape (ncp, nvl, 3)

    r1_norm = np.sqrt(np.sum(r1 ** 2, axis=2))      # r1_norm shape = (ncp, nvl)
    r1_norm = r1_norm.reshape(r1_norm.shape + (1,)) # add 3rd dimension
    r2_norm = np.sqrt(np.sum(r2 ** 2, axis=2))      # r2_norm shape = (ncp, nvl)
    r2_norm = r2_norm.reshape(r2_norm.shape + (1,)) # add 3rd dimension

    cross_r1r2 = np.cross(r1, r2)
    dotr1r2 = np.sum(r1 * r2, axis=2)
    dotr1r2 = dotr1r2.reshape(dotr1r2.shape + (1,)) # add 3rd dimension
    r1r2 = r1_norm * r2_norm

    numer = gamma * (r1_norm + r2_norm) * cross_r1r2
    denom = 4 * math.pi * r1r2 * (r1r2 + dotr1r2) + (0.025 * l0) ** 2
    u_gamma = numer / denom

    return u_gamma
