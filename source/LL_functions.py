import numpy as np
import math


def eval_biot_savart(xcp, vortices):
    # print(query_pts)

    ncps = xcp.shape[0]
    nvrts = len(vortices)

    xnode1 = np.concatenate([obj.node1.reshape(1, -1) for obj in vortices], axis=0)
    xnode2 = np.concatenate([obj.node2.reshape(1, -1) for obj in vortices], axis=0)
    gamma = np.array([obj.circ for obj in vortices]).reshape(1, 1, -1)
    l0 = np.array([obj.length0 for obj in vortices]).reshape(1, 1, -1)
    # print(gamma.shape)

    r1 = xcp.reshape(ncps, 3, 1) - xnode1.reshape(1, 3, nvrts)
    r2 = xcp.reshape(ncps, 3, 1) - xnode2.reshape(1, 3, nvrts)
    # print(r1.shape)

    r1_norm = np.linalg.norm(r1, axis=1, keepdims=True)
    r2_norm = np.linalg.norm(r2, axis=1, keepdims=True)
    # print(r1_norm.shape)

    numer = gamma * (r1_norm + r2_norm) * np.cross(r1, r2, axisa=1, axisb=1, axisc=1)

    r1r2 = r1_norm * r2_norm
    r1r2dot = np.sum(r1 * r2, axis=1, keepdims=True)
    denom = 4 * math.pi * r1r2 * (r1r2 + r1r2dot) + 0.025 * l0

    ucp = numer / denom

    print(ucp)
    print(ucp.shape)
    
    return ucp