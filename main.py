from source.classes import LiftingSurface, FoilAssembly
from source.LL_functions import eval_biot_savart
import numpy as np

from numpy import matlib

# rho*u*L/u
# Seawater properties: https://www.engineeringtoolbox.com/sea-water-properties-d_840.html
# Assumes 15 deg temperature seawater
u = 5           # flow speed in m/s
chord = 0.2     # characteristic length
re = u * chord * 1026 / 0.00126
print(re)

# Instantiate a front wing
front_wing = LiftingSurface(rt_chord=250,
                            tip_chord=80,
                            span=800,
                            sweep_tip=-200,
                            sweep_curv=3,
                            dih_tip=-75,
                            dih_curve=2)
front_wing.generate_coords(npts=101)
front_wing.define_aerofoil('naca2412', False)
front_wing.compute_afoil_polar(angles=np.linspace(-5,15,21), Re=re, plot_flag=False)
# front_wing.plot2D()
# front_wing.plot3D()
print("Front wing area =", str(front_wing.calc_simple_proj_wing_area()))
print("Trapz front wing area =", str(front_wing.calc_trapz_proj_wing_area()))
print("Front wing aspect ratio =", str(front_wing.calc_AR()))
print("Front wing lift =", str(front_wing.calc_lift(V=5, aoa=1, rho=1025)), "Newtons")

front_wing.generate_LL_geom(10)
print("Lifting line front wing area =", str(front_wing.LL_seg_area))

# Instantiate stabiliser
stabiliser = LiftingSurface(rt_chord=90,
                            tip_chord=30,
                            span=400,
                            sweep_tip=-30,
                            sweep_curv=2,
                            dih_tip=30,
                            dih_curve=8)
stabiliser.generate_coords(npts=101)
stabiliser.define_aerofoil('naca0012', False)
stabiliser.compute_afoil_polar(angles=np.linspace(-5, 15, 21), Re=re, plot_flag=False)
# stabiliser.plot2D()
# stabiliser.plot3D()
print("Stabiliser simple wing area =", str(stabiliser.calc_simple_proj_wing_area()))
print("Stabiliser trapz wing area =", str(stabiliser.calc_trapz_proj_wing_area()))
print("Stabiliser aspect ratio =", str(stabiliser.calc_AR()))
print("Stabiliser lift =", str(stabiliser.calc_lift(V=5, aoa=-2, rho=1025)), "Newtons")

# Instantiate a mast
mast = LiftingSurface(rt_chord=130,
                      tip_chord=130,
                      span=600,
                      type='mast')
mast.generate_coords(npts=101)
# mast.plot2D()




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

# to-do:
# - in LiftingSurface class: designate whether a BV is on the LL or not (vtype)
# - write out circulation residual to be solved
# - code up residual, implement some auto-diff of it
