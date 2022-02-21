import numpy as np
import math


# from numba import jit


# @jit(nopython=True)
def eval_biot_savart(xcp, xnode1, xnode2, gamma, l0):
    xcp = xcp.reshape(-1, 1, 3)  # xcp shape (ncp, 1, 3)
    r1 = xcp - xnode1  # r1 shape (ncp, nvl, 3)
    r2 = xcp - xnode2  # r2 shape (ncp, nvl, 3)

    r1_norm = np.sqrt(np.sum(r1 ** 2, axis=2))  # r1_norm shape = (ncp, nvl)
    r1_norm = r1_norm.reshape(r1_norm.shape + (1,))  # add 3rd dimension
    r2_norm = np.sqrt(np.sum(r2 ** 2, axis=2))  # r2_norm shape = (ncp, nvl)
    r2_norm = r2_norm.reshape(r2_norm.shape + (1,))  # add 3rd dimension

    cross_r1r2 = np.cross(r1, r2)
    dotr1r2 = np.sum(r1 * r2, axis=2)
    dotr1r2 = dotr1r2.reshape(dotr1r2.shape + (1,))  # add 3rd dimension
    r1r2 = r1_norm * r2_norm

    numer = gamma * (r1_norm + r2_norm) * cross_r1r2
    denom = 4 * math.pi * r1r2 * (r1r2 + dotr1r2) + (0.025 * l0) ** 2
    u_gamma = numer / denom

    return u_gamma


def lift_from_circulation(rho, gamma, u_cp, dl, a1, a3):
    cross_ucp_dl = np.cross(u_cp, dl)
    L_gamma = rho * gamma * np.sqrt((np.dot(cross_ucp_dl, a1)) ** 2 + (np.dot(cross_ucp_dl, a3)) ** 2)
    return L_gamma # should be size (n_seg,1)


def LL_residual(gamma, rho, u_BV, u_FV, u_motion, dl, a1, a3, cl_spline, dA):

    u_BV = np.sum(u_BV * gamma, axis=1)

    u_cp = u_motion + u_BV + u_FV

    L_gamma = lift_from_circulation(rho, gamma, u_cp, dl, a1, a3)

    L_alpha = lift_from_strip_theory(rho, u_cp, a1, a3, cl_spline, dA)

    R = L_alpha - L_gamma
    return R


def rotation_matrix(w, angle, deg=True):
    # w is the axis to rotate about, size (3,)
    # angle is the angle to rotate by in radian
    if deg == True:
        angle = angle * np.pi / 180
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[w[0] ** 2 * (1 - c) + c, w[0] * w[1] * (1 - c) - w[2] * s, w[0] * w[2] * (1 - c) + w[1] * s, 0],
                  [w[1] * w[0] * (1 - c) + w[2] * s, w[1] ** 2 * (1 - c) + c, w[1] * w[2] * (1 - c) - w[0] * s, 0],
                  [w[2] * w[0] * (1 - c) - w[1] * s, w[2] * w[1] * (1 - c) + w[0] * s, w[2] ** 2 * (1 - c) + c, 0],
                  [0, 0, 0, 1]])

    return R


def apply_rotation(R, vec, dim):
    # R is rotation matrix size (4,4)
    # vec is set of vectors (m,3), or (3,m)
    # dim is the dimension that the vector is in: 0 or 1

    if dim == 0:
        vec = np.vstack((vec, np.ones((1, vec.shape[1]))))
        vec_rot = np.dot(R, vec)
        vec_rot = vec_rot[0:3, :]
    elif dim == 1:
        vec = np.hstack((vec, np.ones((vec.shape[0], 1)))).T
        vec_rot = np.dot(R, vec).T
        vec_rot = vec_rot[:, 0:3]

    return vec_rot


def translation_matrix(t):
    T = np.array([[1, 0, 0, t[0]],
                  [0, 1, 0, t[1]],
                  [0, 0, 1, t[2]],
                  [0, 0, 0, 1]])
    return T
