from source.classes import LiftingSurface, FoilAssembly, ms2knts, knts2ms
from source.LL_functions import eval_biot_savart
import numpy as np
import matplotlib.pyplot as plt

from numpy import matlib

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
front_wing = LiftingSurface(rt_chord=250,
                            tip_chord=80,
                            span=800,
                            Re=re,
                            sweep_tip=-200,
                            sweep_curv=3,
                            dih_tip=-75,
                            dih_curve=2,
                            afoil_name='naca2412')

# Instantiate stabiliser
stabiliser = LiftingSurface(rt_chord=90,
                            tip_chord=50,
                            span=500,
                            Re=re,
                            sweep_tip=-30,
                            sweep_curv=2,
                            dih_tip=30,
                            dih_curve=8,
                            afoil_name='naca0012')

# Instantiate a mast
mast = LiftingSurface(rt_chord=130,
                      tip_chord=130,
                      span=600,
                      Re=re,
                      type='mast',
                      afoil_name='naca0015')  # Axis mast is 19 mm thick, and around 130 mm chord = ~15% thickness

# Assemble the foil
foil = FoilAssembly(front_wing,
                    stabiliser,
                    mast,
                    fuselage_length=699 - 45 - 45,  # assumes AXIS short black fuselage
                    mast_attachment_ratio=267 - 45,  # assumes AXIS short black fuselage
                    wing_angle=1,
                    stabiliser_angle=-2)

foil.rotate_foil_assembly([1, 0, 0])
print(np.sum(foil.compute_foil_loads(u_motion, 1025), axis=0))
foil.rotate_foil_assembly([-1, 0, 0])

angle = np.linspace(-5,10,16)
mom = np.zeros(angle.shape)
for i in range(len(angle)):
    foil.rotate_foil_assembly([angle[i], 0, 0])
    loads = np.sum(foil.compute_foil_loads(u_motion, 1025), axis=0)
    mom[i] = loads[3]
    foil.rotate_foil_assembly([-angle[i], 0, 0])

plt.plot(angle, mom, 'k-')
plt.grid(True)
plt.show()
# foil.plot_foil_assembly()




xnode1 = np.concatenate([obj.node1.reshape(1, 1, -1) for obj in front_wing.BVs], axis=1)
xnode2 = np.concatenate([obj.node2.reshape(1, 1, -1) for obj in front_wing.BVs], axis=1)
gamma = np.array([obj.circ for obj in front_wing.BVs]).reshape(1, -1, 1)
l0 = np.array([obj.length0 for obj in front_wing.BVs]).reshape(1, -1, 1)

xcp = front_wing.xcp

V = eval_biot_savart(xcp, xnode1, xnode2, gamma, l0)
print(V.shape)

# to-do:
# - in LiftingSurface class: designate whether a BV is on the LL or not (vtype)
# - write out circulation residual to be solved
# - code up residual, implement some auto-diff of it
