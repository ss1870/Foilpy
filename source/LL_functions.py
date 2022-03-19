from functools import partial, partialmethod
from re import U
import numpy as np1
import jax.numpy as np
from jax import jit, jacfwd, lax
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
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
def eval_biot_savart(xcp0, xnode1, xnode2, gamma, l0, delta_visc=0.025):
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
    denom = 4 * math.pi * (r1r2 * (r1r2 + dotr1r2) + (delta_visc * l0.reshape(1,-1,1)) ** 2)
    u_gamma = numer / denom
    mask = np.isnan(u_gamma) | np.isinf(u_gamma)
    u_gamma = u_gamma.at[mask].set(0)

    return u_gamma

def ini_estimate_gamma(u_cp, dl, a1, a3, cl_tab, dA, rho):
    # compute lift due to strip theory
    dot_ucp_a1 = np.sum(u_cp * a1, axis=1)
    dot_ucp_a3 = np.sum(u_cp * a3, axis=1)
    alpha_cp = np.arctan2(dot_ucp_a3, dot_ucp_a1)
    # cl = cl_spline.__call__(alpha_cp * 180.0 / np.pi)
    cl = np.interp(alpha_cp * 180.0 / np.pi, cl_tab[:,0], cl_tab[:,1])
    L_alpha = cl * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * np.squeeze(dA)

    # use equation for lift due to circulation to back calculate gamma
    cross_ucp_dl = np.cross(u_cp, dl)
    dot_a1 = np.sum(cross_ucp_dl * a1, axis=1)
    dot_a3 = np.sum(cross_ucp_dl * a3, axis=1)
    gamma = L_alpha/rho/np.sqrt(dot_a1 ** 2 + dot_a3 ** 2) 
    return gamma

def LL_residual(gamma, rho, u_BV, u_FV, u_motion, dl, a1, a3, cl_tab, dA, nseg):
    # gamma:    (nseg,)
    # u_BV:     (nseg, nseg*4, 3)
    # u_motion: (3,)
    # u_FV, dl, a1, a3: (nseg, 3)
    # dA: (nseg,1)
    # cl_spline = callable object to compute lift coefficient
    # cl_tab: (nafoilpolar_pts, 2) lift polar/look-up table
    # nseg: list of number of segments in each lifting surface

    n_surfaces = len(nseg)

    # tile and reshape gamma to be correct size for multiypling by u_BV
    for i in range(n_surfaces):
        if i==0:
            # gamma1 = np.tile(gamma[nseg[i,0]:nseg[i,1]].reshape(1, -1, 1), (len(gamma), 4, 3))
            gamma1 = np.tile(gamma[0:nseg[0]].reshape(1, -1, 1), (len(gamma), 4, 3))
        else:
            # tiled_gamma = np.tile(gamma[nseg[i,0]:nseg[i,1]].reshape(1, -1, 1), (len(gamma), 4, 3))
            tiled_gamma = np.tile(gamma[nseg[0]*(i-1):nseg[0]*i].reshape(1, -1, 1), (len(gamma), 4, 3))
            gamma1 = np.concatenate((gamma1, tiled_gamma), axis=1)

    # multiply gamma * velocity component due to bound vorticity
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
    for i in range(n_surfaces):
        if i==0:
            # cl = cl_tab[i].__call__(alpha_cp[0:50] * 180.0 / np.pi)
            cl = np.interp(alpha_cp[0:nseg[0]] * 180.0 / np.pi, cl_tab[i][:,0], cl_tab[i][:,1])
        else:
            # cl = np.concatenate((cl, cl_tab[i].__call__(alpha_cp[50*(i-1):50*i] * 180.0 / np.pi)))
            cl = np.concatenate((cl, np.interp(alpha_cp[nseg[0]*(i-1):nseg[0]*i] * 180.0 / np.pi, cl_tab[i][:,0], cl_tab[i][:,1])))
    L_alpha = cl * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * np.squeeze(dA)

    # difference between two methods = residual
    R = L_alpha - L_gamma
    return R

def newton_raphson_solver(f, J, x0, nit=1000, tol=1e-7, display=True):
    x = deepcopy(x0)
    xnew = deepcopy(x0)
    if display:
        step_norm_save = np1.zeros((nit))
        R_norm_save = np1.zeros((nit))
    i = 0
    R_norm = 10
    while i < nit and R_norm > tol:
        fnew = f(x)
        xnew = x - np.dot(np.linalg.inv(J(x)), fnew)
        step_norm = np.sqrt(np.sum((xnew-x)**2))
        R_norm = np.sqrt(np.sum(fnew**2))
        if display:
            print("Step ", str(i), " - x_step = ", str(step_norm), " - R_norm = ", str(R_norm))
            step_norm_save[i] = step_norm
            R_norm_save[i] = R_norm

        i = i + 1
        x = xnew

    if display:
        print("Newton-raphson solver finished in ", str(i), " iterations.")
        if R_norm < tol:
            print("Converged due to R-norm being less than tolerance: ", str(tol))
        else:
            print("Converged due to reaching max number of iterations: ", str(nit))
        return x, step_norm_save[0:i], R_norm_save[0:i]
    else:
        return x

def newton_raphson4jit(f, J, x0, tol=1e-4):

    init_val = deepcopy(x0)
    cond_fun = lambda x: np.linalg.norm(f(x)) > tol
    body_fun = lambda x: x - np.dot(np.linalg.inv(J(x)), f(x))

    x = lax.while_loop(cond_fun, body_fun, init_val)
    return x

def steady_LL_solve(lifting_surfaces, u_flow, rho, dt = 0.1, shed_elements_flag = True, nit=10, delta_visc=0.025, wake_from_TE_frac=0.25):
    # lifting surfaces = list of dictionaries. Each dictionary contains the BV locations, unit vectors etc
    
    # unpack lifting surface dictionaries
    n_surf = len(lifting_surfaces)
    for i in range(n_surf):
        if i==0:
            xcp = lifting_surfaces[i]["xcp"]
            a1 = lifting_surfaces[i]["a1"]
            a3 = lifting_surfaces[i]["a3"]
            dA = lifting_surfaces[i]["dA"]
            dl = lifting_surfaces[i]["dl"]
            # cl_spl = [lifting_surfaces[i]["cl_spl"]]
            cl_tab = [lifting_surfaces[i]["cl_tab"]]
            xnode1 = lifting_surfaces[i]["xnode1"]
            xnode2 = lifting_surfaces[i]["xnode2"]
            TE = [lifting_surfaces[i]["TE"]]
            l0 = lifting_surfaces[i]["l0"]
            # n_seg = np.array([0, xcp.shape[0]]).reshape(1,-1)
            n_seg = [xcp.shape[0]]
            gamma_ini = ini_estimate_gamma(u_flow, dl, a1, a3, cl_tab[0], dA, rho)
        else:
            xcp = np.vstack((xcp, lifting_surfaces[i]["xcp"]))
            a1 = np.vstack((a1, lifting_surfaces[i]["a1"]))
            a3 = np.vstack((a3, lifting_surfaces[i]["a3"]))
            dA = np.vstack((dA, lifting_surfaces[i]["dA"]))
            dl = np.vstack((dl, lifting_surfaces[i]["dl"]))
            xnode1 = np.concatenate((xnode1, lifting_surfaces[i]["xnode1"]), axis=1)
            xnode2 = np.concatenate((xnode2, lifting_surfaces[i]["xnode2"]), axis=1)
            l0 = np.concatenate((l0, lifting_surfaces[i]["l0"]))
            # cl_spl.append(lifting_surfaces[i]["cl_spl"])
            cl_tab.append(lifting_surfaces[i]["cl_tab"])
            TE.append(lifting_surfaces[i]["TE"])
            # n_seg = np.concatenate((n_seg, np.array([n_seg[i-1,1], n_seg[i-1,1]+lifting_surfaces[i]["xcp"].shape[0]]).reshape(1,-1)), axis=0)
            n_seg.append(lifting_surfaces[i]["xcp"].shape[0])
            gamma_ini = np.concatenate((gamma_ini, ini_estimate_gamma(u_flow, 
                                                        lifting_surfaces[i]["dl"], 
                                                        lifting_surfaces[i]["a1"], 
                                                        lifting_surfaces[i]["a3"], 
                                                        lifting_surfaces[i]["cl_tab"], 
                                                        lifting_surfaces[i]["dA"],
                                                        rho)))

    # pre-allocate wake element table
    # description of wake element table structure:
    # - all elements stemming from 1st surface, then 2nd surface, then 3rd, and so on...
    # - for one surface, we have 4*n_seg elements at t=0, and then 3*n_seg elements from then on
    # - for t=0, create a new full ring, with same order as for the BVs: LL(front), RHS, TE(rear), LHS
    # - for t>0, create only the front of a ring, with order: LL(front), RHS, LHS
    n_elmts_layer1 = sum(n_seg)*4
    n_elmts_all_other_layers = sum(n_seg)*3
    n_wake_elmts = n_elmts_layer1 + nit * n_elmts_all_other_layers
    # if shed_elements_flag==True:
        # n_elmts_layer1 = 4*
        # n_elmts_per_layer = 
    # wake_elmt_table: [xnode1 (nelmts, 3), xnode2 (nelmts, 3), gamma (nelmts,), l0 (nelmts,)]
    wake_elmt_table = np1.zeros((n_wake_elmts, 8))
    convectable_elmts_all = -np1.ones((n_wake_elmts, 2), np1.int8)
    elmtIDs_all = -np1.ones((n_wake_elmts, 3), np1.int64)
    nFVs = 0
    gamma_BVs = gamma_ini
    gamma_BV_step = np1.zeros((nit))
    # do the quasi-time loop
    for t in range(nit):
        # eval biot-savart at CPs due to bound vorticity
        u_BV = eval_biot_savart(xcp, xnode1, xnode2, np.ones((l0.shape)), l0, delta_visc=delta_visc)
        # eval biot-savart at CPs due to free wake vorticity
        u_FV = np.zeros((1,3))
        if nFVs > 0:
            u_FV = eval_biot_savart(xcp, wake_elmt_table[0:nFVs, 0:3], wake_elmt_table[0:nFVs, 3:6], wake_elmt_table[0:nFVs, 6], wake_elmt_table[0:nFVs, 7], delta_visc=delta_visc)        
            u_FV = np.sum(u_FV, axis=1)

        # calc circulation at lifting surfaces
        # declare residual function
        R = lambda gamma: LL_residual(gamma, rho, u_BV, u_FV, u_flow, dl, a1, a3, cl_tab, dA, n_seg)
        J = jacfwd(R) # declare jacobian of residual

        # compute circulation
        gamma_BVs_prev = gamma_BVs
        gamma_BVs, step, R = newton_raphson_solver(R, J, gamma_BVs, nit=100, tol=1e-4, display=True)
        gamma_BV_step[t] = np.sqrt(np.sum((gamma_BVs - gamma_BVs_prev) ** 2))
        # gamma_BVs = newton_raphson4jit(R, J, gamma_ini, tol=1e-4)
        gamma_elmt = np.empty((0))
        for i in range(n_surf):
            gamma_elmt = np.concatenate((gamma_elmt, np.tile(gamma_BVs[n_seg[0]*i:n_seg[0]*(i+1)], (4))))


        # convect/update wake
        if nFVs > 0:
            # calc velocities at wake nodes due to u_gamma and u_flow
            # get unique wake nodes
            # unique_wake_nodes = np.unique(np.vstack((wake_elmt_table[0:nFVs, 0:3], wake_elmt_table[0:nFVs, 3:6])), axis=0)
            # # compute induced velocity at wake nodes due to free vorticity
            # u_wake_FV = eval_biot_savart(unique_wake_nodes, wake_elmt_table[0:nFVs, 0:3], wake_elmt_table[0:nFVs, 3:6], wake_elmt_table[0:nFVs, 6], wake_elmt_table[0:nFVs, 7], delta_visc)
            # u_wake_FV = np.sum(u_wake_FV, axis=1)
            # # compute induced velocity at wake nodes due to bound vorticity
            # u_wake_BV = eval_biot_savart(unique_wake_nodes, xnode1, xnode2, gamma_elmt, l0, delta_visc)
            # u_wake_BV = np.sum(u_wake_BV, axis=1)

            u_convect = u_flow # + u_wake_FV + u_wake_BV

            # convect wake nodes 
            # convect node 1's
            mask = convectable_elmts_all[:,0]==1
            wake_elmt_table[mask, 0:3] = wake_elmt_table[mask, 0:3] + u_convect*dt
            # convext node 2's
            mask = convectable_elmts_all[:,1]==1
            wake_elmt_table[mask, 3:6] = wake_elmt_table[mask, 3:6] + u_convect*dt

            # convect TE nodes to fractionally convected TE
            # convect node 1's
            mask = convectable_elmts_all[:,0]==0
            wake_elmt_table[mask, 0:3] = wake_elmt_table[mask, 0:3] + wake_from_TE_frac * u_flow * dt
            convectable_elmts_all[mask,0] = 1 # update these nodes are now convectable
            # convect node 2's
            mask = convectable_elmts_all[:,1]==0
            wake_elmt_table[mask, 3:6] = wake_elmt_table[mask, 3:6] + wake_from_TE_frac * u_flow * dt
            convectable_elmts_all[mask,1] = 1 # update these nodes are now convectable


        # add new wake elements - add elmt connectivity, node locations, and gamma
        # include option to ignore shed elements?
        near_wake_elmts, convectable_elmts, elmtIDs, n_FVs_new = add_wake_elmts(n_surf, TE, wake_from_TE_frac, u_flow, dt, gamma_BVs, n_seg, nFVs, t)
        wake_elmt_table[nFVs:n_FVs_new, 0:7] = near_wake_elmts
        convectable_elmts_all[nFVs:n_FVs_new, 0:2] = convectable_elmts
        elmtIDs_all[nFVs:n_FVs_new, 0:3] = elmtIDs
        nFVs = n_FVs_new
        # upate gamma of previous layer (avoids creating duplicate elmts on same line)
        if t > 0: # and include_shed_vorticity==True:
            mask = (elmtIDs_all[:, 0] == t - 1) & (elmtIDs_all[:, 1] == 3)
            wake_elmt_table[mask, 6] = wake_elmt_table[mask, 6] - gamma_BVs

        # update element lengths
        wake_elmt_table[0:nFVs, 7] = update_elmt_length(wake_elmt_table[0:nFVs, 0:3], wake_elmt_table[0:nFVs, 3:6]) 

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(np.vstack((wake_elmt_table[0:nFVs, 0], wake_elmt_table[0:nFVs, 3])), 
                np.vstack((wake_elmt_table[0:nFVs, 1], wake_elmt_table[0:nFVs, 4])), 
                np.vstack((wake_elmt_table[0:nFVs, 2], wake_elmt_table[0:nFVs, 5])))
    plt.show()

    u_BV = np.sum(eval_biot_savart(xcp, xnode1, xnode2, gamma_elmt, l0, delta_visc=delta_visc), axis=1)
    u_FV = np.sum(eval_biot_savart(xcp, wake_elmt_table[0:nFVs, 0:3], wake_elmt_table[0:nFVs, 3:6], wake_elmt_table[0:nFVs, 6], wake_elmt_table[0:nFVs, 7], delta_visc=delta_visc), axis=1)     
    u_cp = u_BV + u_FV

    return u_cp, gamma_ini, gamma_BVs, wake_elmt_table 

def add_wake_elmts(n_surf, TE, wake_from_TE_frac, u_flow, dt, gamma_BVs, n_seg, nFVs, t, include_shed_vorticity=True):
    new_wake_elmts = np.empty((0,7))
    convectable_elmts = np.empty((0,2))
    elmtIDs = np.empty((0,3))
    for i in range(n_surf):
        TE_nodes = TE[i]
        first_shed_nodes = TE[i] + wake_from_TE_frac * u_flow * dt
        gamma_BVs_i = gamma_BVs[n_seg[i]*i:n_seg[i]*(i+1)]
        gamma_trail = np1.zeros((n_seg[i]+1, 1))
        gamma_trail[0:n_seg[i],0] = -gamma_BVs_i
        gamma_trail[1:,0] = gamma_trail[1:,0] + gamma_BVs_i
        shed_elmts = np.empty((0,7))
        shed_elmts1 = np.empty((0,7))
        if include_shed_vorticity:
            if t==0:
                # create n_seg shed elmts at frac_conv_TE
                # only released on first time step, forms final line of wake elements in vortex lattice
                shed_elmts1 = np.hstack((first_shed_nodes[1:,:], first_shed_nodes[0:-1,:], gamma_BVs_i.reshape(-1,1)))
            # create n_seg shed elmts at TE
            shed_elmts = np.hstack((TE_nodes[0:-1,:], TE_nodes[1:,:], gamma_BVs_i.reshape(-1,1)))

        # create n_seg+1 trailing elements
        trailing_elmts = np.hstack((TE_nodes, first_shed_nodes, gamma_trail))
        
        # assemble outputs: new_wake_elmts, convectable_elmts, elmtIDs
        new_wake_elmts = np.vstack((new_wake_elmts, 
                                    shed_elmts1,       # 1. only released on first time step
                                    trailing_elmts,    # 2. trailing elements
                                    shed_elmts))       # 3. shed elements at TE
        convectable_elmts = np.vstack((convectable_elmts, 
                                       np.ones((shed_elmts1.shape[0],2)),           # 1. shed elmts 1
                                       np.array([[0,1]]*(trailing_elmts.shape[0])), # 2. trailing elements
                                       np.zeros((shed_elmts.shape[0],2))))          # 3. shed elements at TE
        elmtIDs = np.vstack((elmtIDs, 
                             np.tile(np.array([[t, 1, i]]), (shed_elmts1.shape[0], 1)),       # 1. shed elmts 1
                             np.tile(np.array([[t, 2, i]]), (trailing_elmts.shape[0], 1)),    # 2. trailing elements
                             np.tile(np.array([[t, 3, i]]), (shed_elmts.shape[0], 1))))       # 3. shed elements at TE
                                           
    n_FVs_new = nFVs + new_wake_elmts.shape[0]
    return new_wake_elmts, convectable_elmts, elmtIDs, n_FVs_new


def update_elmt_length(nodes1, nodes2):
    length = np.sqrt(np.sum((nodes2-nodes1) ** 2, axis=1))
    return length

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
