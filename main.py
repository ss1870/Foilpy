from source.classes import LiftingSurface, FoilAssembly
from source.LL_functions import eval_biot_savart
import numpy as np

from numpy import matlib

# rho*u*L/u
# Seawater properties: https://www.engineeringtoolbox.com/sea-water-properties-d_840.html
# Assumes 15 deg temperature seawater
u = 5  # flow speed in m/s
chord = 0.2  # characteristic length
re = u * chord * 1026 / 0.00126
print(re)

# Instantiate a front wing
front_wing = LiftingSurface(rt_chord=250,
                            tip_chord=80,
                            span=800,
                            Re=re,
                            sweep_tip=-200,
                            sweep_curv=3,
                            dih_tip=-75,
                            dih_curve=2,
                            afoil_name='naca2412')

u_motion = np.array([0, 5, 0]).reshape(1, 3)
forces = front_wing.LL_strip_theory_forces(u_motion, 1025)
print(np.sum(forces, axis=0))
print("Lifting line front wing area =", str(front_wing.LL_seg_area))

# Instantiate stabiliser
stabiliser = LiftingSurface(rt_chord=90,
                            tip_chord=30,
                            span=400,
                            Re=re,
                            sweep_tip=-30,
                            sweep_curv=2,
                            dih_tip=30,
                            dih_curve=8,
                            afoil_name='naca0012')
u_motion = np.array([0, 5, 0]).reshape(1, 3)
forces = stabiliser.LL_strip_theory_forces(u_motion, 1025)
print(np.sum(forces, axis=0))
print("Lifting line front wing area =", str(stabiliser.LL_seg_area))

# Instantiate a mast
mast = LiftingSurface(rt_chord=130,
                      tip_chord=130,
                      span=600,
                      Re=re,
                      type='mast',
                      afoil_name='naca0015') # Axis mast is 19 mm thick, and around 130 mm chord
u_motion = np.array([0, 5, 0]).reshape(1, 3)
forces = mast.LL_strip_theory_forces(u_motion, 1025)
print(np.sum(forces, axis=0))
print("Lifting line front wing area =", str(mast.LL_seg_area))


# foil = FoilAssembly(front_wing,
#                     stabiliser,
#                     mast,
#                     fuselage_length=699 - 45 - 45,      # assumes AXIS short black fuselage
#                     mast_attachment_ratio=267 - 45,     # assumes AXIS short black fuselage
#                     wing_angle=1,
#                     stabiliser_angle=-2)


# xnode1 = np.concatenate([obj.node1.reshape(1, 1, -1) for obj in front_wing.BVs], axis=1)
# xnode2 = np.concatenate([obj.node2.reshape(1, 1, -1) for obj in front_wing.BVs], axis=1)
# gamma = np.array([obj.circ for obj in front_wing.BVs]).reshape(1, -1, 1)
# l0 = np.array([obj.length0 for obj in front_wing.BVs]).reshape(1, -1, 1)
#
# xcp = front_wing.xcp
#
# V = eval_biot_savart(xcp, xnode1, xnode2, gamma, l0)

# - why is mast producing non-zero lift?
# - implement assemble foil, i.e. rotate/translate LL arrays accordingly
# - then can compute forces on all bodies in their final orientation and get forces/moments on overall assembly

# to-do:
# - in LiftingSurface class: designate whether a BV is on the LL or not (vtype)
# - write out circulation residual to be solved
# - code up residual, implement some auto-diff of it
