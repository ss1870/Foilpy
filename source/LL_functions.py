import numpy as np1
import jax.numpy as np
from jax import jacfwd
import math
from copy import deepcopy


# from numba import jit

def numerical_jacobian(f, x, h=1e-4):
    J = np1.zeros((len(x), len(x)))
    for i in range(len(x)):
        xph = deepcopy(x)
        xph[i] = x[i] + h
        xmh = deepcopy(x)
        xmh[i] = xmh[i] - h
        J[:,i] = (f(xph) - f(xmh))/2/h
    return J

# @jit(nopython=True)
def eval_biot_savart(xcp0, xnode1, xnode2, gamma, l0):
    # xcp: (ncp, 3)
    # xnode1, xnode2: (1, nvor, 3)
    # gamma, l0: (nvor,)

    xcp = xcp0.reshape(-1, 1, 3)  # xcp shape (ncp, 1, 3)
    # dim [1] of xcp is broadcast nvor times
    # dim [0] of xnode1/2 is broadcast ncp times
    r1 = xcp - xnode1  # r1 shape (ncp, nvor, 3)
    r2 = xcp - xnode2  # r2 shape (ncp, nvor, 3)

    r1_norm = np.sqrt(np.sum(r1 ** 2, axis=2))  # r1_norm shape = (ncp, nvl)
    r1_norm = r1_norm.reshape(r1_norm.shape + (1,))  # add 3rd dimension
    r2_norm = np.sqrt(np.sum(r2 ** 2, axis=2))  # r2_norm shape = (ncp, nvl)
    r2_norm = r2_norm.reshape(r2_norm.shape + (1,))  # add 3rd dimension

    cross_r1r2 = np.cross(r1, r2)
    dotr1r2 = np.sum(r1 * r2, axis=2)
    dotr1r2 = dotr1r2.reshape(dotr1r2.shape + (1,))  # add 3rd dimension
    r1r2 = r1_norm * r2_norm

    numer = gamma.reshape(1,-1,1) * (r1_norm + r2_norm) * cross_r1r2
    denom = 4 * math.pi * r1r2 * (r1r2 + dotr1r2) + (0.025 * l0.reshape(1,-1,1)) ** 2
    u_gamma = numer / denom

    return u_gamma

def ini_estimate_gamma(u_cp, dl, a1, a3, cl_spline, dA, rho):
    # compute lift due to strip theory
    dot_ucp_a1 = np.sum(u_cp * a1, axis=1)
    dot_ucp_a3 = np.sum(u_cp * a3, axis=1)
    alpha_cp = np.arctan2(dot_ucp_a3, dot_ucp_a1)
    cl = cl_spline.__call__(alpha_cp * 180.0 / np.pi)
    L_alpha = cl * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * np.squeeze(dA)

    # use equation for lift due to circulation to back calculate gamma
    cross_ucp_dl = np.cross(u_cp, dl)
    dot_a1 = np.sum(cross_ucp_dl * a1, axis=1)
    dot_a3 = np.sum(cross_ucp_dl * a3, axis=1)
    gamma = L_alpha/rho/np.sqrt(dot_a1 ** 2 + dot_a3 ** 2) 
    return gamma

def LL_residual(gamma, rho, u_BV, u_FV, u_motion, dl, a1, a3, cl_spline, dA, nseg):
    # gamma:    (nseg,)
    # u_BV:     (nseg, nseg*4, 3)
    # u_motion: (3,)
    # u_FV, dl, a1, a3: (nseg, 3)
    # dA: (nseg,1)
    # cl_spline = callable object to compute lift coefficient

    # multiply gamma * velocity component due to bound vorticity
    gamma1 = np.tile(gamma.reshape(1, -1, 1), (len(gamma), 4, 3))
    u_BV = np.sum(u_BV * gamma1, axis=1)  # resulting size (nseg, 3)

    # sum up all velocity components at the CPs
    u_cp = u_motion.reshape(1, 3) + u_BV + u_FV  # (nseg, 3)

    # compute lift due to circulation
    cross_ucp_dl = np.cross(u_cp, dl)
    dot_a1 = np.sum(cross_ucp_dl * a1, axis=1)
    dot_a3 = np.sum(cross_ucp_dl * a3, axis=1)
    L_gamma = rho * gamma * np.sqrt(dot_a1 ** 2 + dot_a3 ** 2) # (nseg,)

    # compute lift due to strip theory
    dot_ucp_a1 = np.sum(u_cp * a1, axis=1)
    dot_ucp_a3 = np.sum(u_cp * a3, axis=1)
    alpha_cp = np.arctan2(dot_ucp_a3, dot_ucp_a1)
    cl = cl_spline.__call__(alpha_cp * 180.0 / np.pi)
    L_alpha = cl * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * np.squeeze(dA)

    # difference between two methods = residual
    R = L_alpha - L_gamma
    return R

def newton_raphson_solver(f, J, x0, nit=1000, tol=1e-7, display=True):
    x = deepcopy(x0)
    xnew = deepcopy(x0)
    step_norm = np1.zeros((nit))
    R_norm = np1.zeros((nit))
    converged_flag = False
    for i in range(nit):
        fnew = f(x)
        xnew = x - np.dot(np.linalg.inv(J(x)), fnew)
        # res[i] = np1.sqrt(np1.sum(fi**2))
        step_norm[i] = np1.sqrt(np1.sum((xnew-x)**2))
        R_norm[i] = np1.sqrt(np1.sum(fnew**2))
        if display:
            print("Step ", str(i), " - x_step = ", str(step_norm[i]), " - R_norm = ", str(R_norm[i]))
        if step_norm[i] < tol:
            converged_flag = True
            i=i+1
            break
        x = xnew

    if display:
        print("Newton-raphson solver finished in ", str(i), " iterations.")
        if converged_flag:
            print("Converged due to step-norm being less than tolerance: ", str(tol))
        else:
            print("Converged due to reaching max number of iterations: ", str(nit))

    return x, step_norm[0:i], R_norm[0:i]

def steady_LL_solve(lifting_surfaces, u_flow, rho, nit=1):
    # lifting surfaces = list of dictionaries? Each dictionary contains the BV locations, unit vectors etc
    
    # unpack lifting surface dictionaries
    n_surf = len(lifting_surfaces)
    for i in range(n_surf):
        if i==0:
            xcp = lifting_surfaces[i]["xcp"]
            dl = lifting_surfaces[i]["dl"]
            a1 = lifting_surfaces[i]["a1"]
            a3 = lifting_surfaces[i]["a3"]
            dA = lifting_surfaces[i]["dA"]
            dl = lifting_surfaces[i]["dl"]
            cl_spl = lifting_surfaces[i]["cl_spl"]
            xnode1 = lifting_surfaces[i]["xnode1"]
            xnode2 = lifting_surfaces[i]["xnode2"]
            l0 = lifting_surfaces[i]["l0"]
            n_seg = xcp.shape[0]
            gamma_ini = ini_estimate_gamma(u_flow, dl, a1, a3, cl_spl, dA, rho)
        else:
            xcp = np.vstack((xcp, lifting_surfaces[i]["xcp"]))
            dl = np.vstack((dl, lifting_surfaces[i]["dl"]))
            a1 = np.vstack((a1, lifting_surfaces[i]["a1"]))
            a3 = np.vstack((a3, lifting_surfaces[i]["a3"]))
            dA = np.vstack((dA, lifting_surfaces[i]["dA"]))
            dl = np.vstack((dl, lifting_surfaces[i]["dl"]))
            xnode1 = np.concatenate((xnode1, lifting_surfaces[i]["xnode1"]), axis=1)
            xnode2 = np.concatenate((xnode2, lifting_surfaces[i]["xnode2"]), axis=1)
            l0 = np.concatenate((l0, lifting_surfaces[i]["l0"]))
            cl_spl = [cl_spl, lifting_surfaces[i]["cl_spl"]]
            n_seg = [n_seg, lifting_surfaces[i]["xcp"].shape[0]]
            gamma_ini = [gamma_ini, ini_estimate_gamma(u_flow, 
                                                        lifting_surfaces[i]["dl"], 
                                                        lifting_surfaces[i]["a1"], 
                                                        lifting_surfaces[i]["a3"], 
                                                        lifting_surfaces[i]["cl_spl"], 
                                                        lifting_surfaces[i]["dA"],
                                                        rho)]

    for i in range(1):
        a=np.array(3)
        # eval biot-savart at CPs due to bound vorticity
        gamma = np.ones((l0.shape))
        u_BV = eval_biot_savart(xcp, xnode1, xnode2, np.ones((l0.shape)), l0)
        # eval biot-savart at CPs due to free wake vorticity
        # u_FV = eval_biot_savart(xcp, xnode1, xnode2, gamma, l0)
        u_FV = np.zeros((1,3))

        # calc circulation at lifting surfaces
        # declare residual function
        # R = lambda gamma: LL_residual(gamma, rho, u_BV, u_FV, u_flow, dl, a1, a3, cl_spl, dA, n_seg)
    #     # declare jacobian of residual
    #     J = jacfwd(R)
    #     # compute circulation
    #     gamma, res = newton_raphson_solver(R, J, gamma, nit=10)

    #     # convect/update wake
    #         # requires eval BS at wake nodes

    # gamma = np.zeros((xcp.shape[0]))
    return gamma_ini #np.sum(u_BV,axis=1) #, u_cp



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

    if dim == -1:
        vec = np.vstack((vec.reshape(3,-1), np.ones((1, 1))))
        vec_rot = np.dot(R, vec)
        vec_rot = vec_rot[0:3, :].reshape(-1)
    elif dim == 0:
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
