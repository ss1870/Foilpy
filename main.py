#%%

from source.classes import LiftingSurface, FoilAssembly, ms2knts, knts2ms
from source.LL_functions import eval_biot_savart, LL_residual, ini_estimate_gamma, newton_raphson_solver, numerical_jacobian, steady_LL_solve
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd
from scipy.optimize import fsolve
# %matplotlib widget

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
                            dih_tip=-0,  # -75
                            dih_curve=2,  # 0
                            afoil_name='naca2412',
                            nsegs=40,
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
                            nsegs=40,
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
print(foil.compute_foil_loads(-u_motion, rho))
u_cp, gamma_ini, gamma_BVs, wake_elmt_table, gamma_hist = steady_LL_solve(lifting_surfaces, -u_motion, rho, dt=0.1, nit=20)
fig = plt.figure()
plt.plot(foil.main_wing.xcp[:,0], gamma_ini[0:foil.main_wing.nsegs], 'r-')
plt.plot(foil.main_wing.xcp[:,0], gamma_BVs[0:foil.main_wing.nsegs], 'g-')
plt.grid(True)
plt.show(block=True)

print(foil.compute_foil_loads(-u_motion, rho, u_cp))

print(str(np.sum(foil.main_wing.LL_strip_theory_forces(-u_motion, rho), axis=0)))
np.sum(foil.main_wing.LL_strip_theory_forces(-u_motion+u_cp[0:foil.main_wing.nsegs], rho), axis=0)
np.sum(foil.stabiliser.LL_strip_theory_forces(-u_motion+u_cp[foil.main_wing.nsegs:(foil.main_wing.nsegs+foil.stabiliser.nsegs)], rho), axis=0)


# foil.rotate_foil_assembly([1, 0, 0])
# print(np.sum(foil.compute_foil_loads(u_motion, 1025), axis=0))
# foil.rotate_foil_assembly([-1, 0, 0])

angle = np.linspace(-5,10,8)
load_save = np.zeros((angle.shape[0], 3))
for i in range(len(angle)):
    foil.rotate_foil_assembly([angle[i], 0, 0])
    loads = foil.compute_foil_loads(-u_motion, rho)
    load_save[i,:] = loads[1:4]
    foil.rotate_foil_assembly([-angle[i], 0, 0])

fig = plt.figure()
plt.plot(angle, load_save, 'k-')
plt.grid(True)
plt.show()




# To-do:
#  - Considerations:
    # - modelling the wing tips accurately and cleanly is important. Be careful with interpolation schemes, and 
    # how well the segment discretisation matches the intended geometry.

# - Tests:
    # - write test to check jax auto diff of residual is working
    # - write test of root finding algorithm
    # - finish elliptical wing test, are there any other things I can test from that?
        # - Get auto test in github?

# - Coding:
    # - Write method to loop through foil pitch angles and compute forces/moments

# - Plotting wake geometry
# - wake convection due to induced flow, and functionalise

# - Unanswered questions:
    # - does large dt make for lower accuracy?
    # - should ommitting shed vortex elements result in same answer?



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
