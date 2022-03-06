#%%
from source.classes import LiftingSurface, FoilAssembly, ms2knts, knts2ms
from source.LL_functions import eval_biot_savart, LL_residual, ini_estimate_gamma, newton_raphson_solver, numerical_jacobian, steady_LL_solve
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd
from scipy.optimize import fsolve

# rho*u*L/u
# Seawater properties: https://www.engineeringtoolbox.com/sea-water-properties-d_840.html
# Assumes 15 deg temperature seawater
u = 5  # flow speed in m/s
chord = 0.2  # characteristic length
re = u * chord * 1026 / 0.00126
print("Reynolds number = ", str(re), "\n")

# Define simple motion vector, moving forward in y at 10 knots
u_motion = np.array([0, knts2ms(10), 0]).reshape(1, 3)
print("Input velocity vector = ", str(u_motion), " m/s (= ", str(ms2knts(u_motion)), " knots)\n ")

# Instantiate a front wing
front_wing = LiftingSurface(rt_chord=200, #250
                            tip_chord=80, #80
                            span=800,
                            Re=re,
                            sweep_tip=-200, #-200
                            sweep_curv=3,   # 3
                            dih_tip=-75,  # -75
                            dih_curve=2,  # 0
                            afoil_name='naca2412',
                            nsegs=50,
                            units='mm')
front_wing.plot2D()

# Instantiate stabiliser
stabiliser = LiftingSurface(rt_chord=90,
                            tip_chord=50,
                            span=500,
                            Re=re,
                            sweep_tip=-30,
                            sweep_curv=2,
                            dih_tip=30,
                            dih_curve=8,
                            afoil_name='naca0012',
                            nsegs=50,
                            units='mm')

# Instantiate a mast
mast = LiftingSurface(rt_chord=130,
                      tip_chord=130,
                      span=600,
                      Re=re,
                      type='mast',
                      afoil_name='naca0015',
                      nsegs=4,
                      units='mm')  # Axis mast is 19 mm thick, and around 130 mm chord = ~15% thickness

# Assemble the foil
foil = FoilAssembly(front_wing,
                    stabiliser,
                    mast,
                    fuselage_length=699 - 45 - 45,  # assumes AXIS short black fuselage
                    mast_attachment_ratio=267 - 45,  # assumes AXIS short black fuselage
                    wing_angle=1,
                    stabiliser_angle=-2,
                    units='mm')

lifting_surfaces = foil.surface2dict()
rho = 1025
# foil.plot_foil_assembly()
# print(np.sum(foil.compute_foil_loads(-u_motion, 1025), axis=0))
steady_LL_solve(lifting_surfaces, -u_motion, rho, nit=1)
a = jit(steady_LL_solve)
out = a(lifting_surfaces, -u_motion, rho, nit=1)
print(out)
# foil.rotate_foil_assembly([1, 0, 0])
# print(np.sum(foil.compute_foil_loads(u_motion, 1025), axis=0))
# foil.rotate_foil_assembly([-1, 0, 0])

# angle = np.linspace(-5,10,16)
# mom = np.zeros(angle.shape)
# for i in range(len(angle)):
#     foil.rotate_foil_assembly([angle[i], 0, 0])
#     loads = np.sum(foil.compute_foil_loads(u_motion, 1025), axis=0)
#     mom[i] = loads[3]
#     foil.rotate_foil_assembly([-angle[i], 0, 0])

# plt.plot(angle, mom, 'k-')
# plt.grid(True)
# plt.show()
# foil.plot_foil_assembly()

# foil.rotate_foil_assembly([1, 0, 0])
# foil.main_wing.generate_LL_geom(50)

# xnode1 = np.concatenate([obj.node1.reshape(1, 1, -1) for obj in foil.main_wing.BVs], axis=1) # (1, nseg*4, 3)
# xnode2 = np.concatenate([obj.node2.reshape(1, 1, -1) for obj in foil.main_wing.BVs], axis=1) # (1, nseg*4, 3)
# # gamma = np.array([obj.circ for obj in front_wing.BVs]).reshape(1, -1, 1)
# l0 = np.array([obj.length0 for obj in foil.main_wing.BVs])
# gamma = np.ones(l0.shape)
# xcp = foil.main_wing.xcp
# u_BV = eval_biot_savart(xcp, xnode1, xnode2, gamma, l0)
# print(u_BV)
# fast_BS = jit(eval_biot_savart)
# u_BV1 = fast_BS(xcp, xnode1, xnode2, gamma, l0)
# print(np.max(u_BV-u_BV1))
# print(u_BV.shape)



# u_FV = jnp.zeros((1,3))
# # R = LL_residual(gamma_ini, rho, u_BV, u_FV, u_motion, foil.main_wing.dl, foil.main_wing.a1, foil.main_wing.a3, foil.main_wing.cl_spline, foil.main_wing.dA)
# f = lambda gamma: LL_residual(gamma, rho, u_BV, u_FV, -u_motion, foil.main_wing.dl, foil.main_wing.a1, foil.main_wing.a3, foil.main_wing.cl_spline, foil.main_wing.dA)
# # # J = jacfwd(f)(gamma_ini)

# gamma_ini = ini_estimate_gamma(-u_motion, foil.main_wing.dl, foil.main_wing.a1, foil.main_wing.a3, foil.main_wing.cl_spline, foil.main_wing.dA, rho)
# gamma_root, step_hist, f_hist = newton_raphson_solver(f, jacfwd(f), gamma_ini, nit=100, tol=1e-5)
# # print(np.sum(u_BV,axis=1))
# # print(gamma_root.reshape(-1,1)*np.sum(u_BV,axis=1))
# print(gamma_root)
# gamma_root1, infodict, ier, mesg = fsolve(f, gamma_ini, fprime=jacfwd(f), full_output=True, col_deriv=False)
# plt.plot(foil.main_wing.xcp[:,0], gamma_ini, 'r-')
# plt.plot(foil.main_wing.xcp[:,0], gamma_root, 'g-')
# plt.plot(foil.main_wing.xcp[:,0], gamma_root1, 'g-')
# plt.grid(True)
# plt.show(block=True)

# print(infodict["fvec"], np.linalg.norm(infodict["fvec"]))
# print(mesg)


# to-do:
# - in LiftingSurface class: designate whether a BV is on the LL or not (vtype)
# - write test to check auto diff of residual is working
# - Work out structure of full LL + wake solver
    # - maybe first try get what I currently have working, i.e. BS for BVs of 1 surface and a calc circulation, JIT'ed within the steady solver function?
        # - need to make LL_residual work for multiple lifting surfaces
    # - Get circulation solver working for multiple lifting surfaces
    # - include wake generation and convection
    # - ensure u_flow and u_motion are differentiated/implemented correctly
    # - ensure all length units are in meters, this is the convention


# Test auto-diff jacobian
# u_FV = np.zeros((1,3))
# f = lambda gamma: LL_residual(gamma, rho, u_BV, u_FV, u_motion, front_wing.dl, front_wing.a1, front_wing.a3, front_wing.cl_spline, front_wing.dA)
# J1 = numerical_jacobian(f, np.array(gamma_ini), 1e-4)
# print(J1)
# print(J1.shape)
# print(J-J1)
# print(np.max(J - J1))

## Test root finding algo with numerical derivative
# u_FV = np.zeros((1,3))
# f = lambda gamma: LL_residual(gamma, rho, u_BV, u_FV, u_motion, front_wing.dl, front_wing.a1, front_wing.a3, front_wing.cl_spline, front_wing.dA)
# J1 = lambda gamma: numerical_jacobian(f, np.array(gamma), 1e-4)
# gamma_root, res = newton_raphson_solver(f, J1, np.array(gamma_ini), nit=10)


# %%
