from source.classes import LiftingSurface, FoilAssembly
from source.LL_functions import eval_biot_savart, eval_biot_savart1
import numpy as np
from numpy import matlib

# Instantiate a front wing
front_wing = LiftingSurface(rt_chord=250,
                            tip_chord=80,
                            span=800,
                            sweep_tip=-200,
                            sweep_curv=3,
                            dih_tip=-75,
                            dih_curve=2)
front_wing.generate_coords(npts=101)
# front_wing.plot2D()
# front_wing.plot3D()
print("Front wing area =", str(front_wing.calc_simple_proj_wing_area()))
print("Trapz front wing area =", str(front_wing.calc_trapz_proj_wing_area()))
print("Front wing aspect ratio =", str(front_wing.calc_AR()))
print("Front wing lift =", str(front_wing.calc_lift(V=5, aoa=4, rho=1025)), "Newtons")

front_wing.generate_LL_geom(10)
print("Lifting line front wing area =", str(front_wing.LL_seg_area))

# Instantiate stabiliser
stabiliser = LiftingSurface(rt_chord=120,
                            tip_chord=30,
                            span=360,
                            sweep_tip=-50,
                            sweep_curv=2,
                            dih_tip=35,
                            dih_curve=4)
stabiliser.generate_coords(npts=101)
# stabiliser.plot2D()
# stabiliser.plot3D()
print("Stabiliser simple wing area =", str(stabiliser.calc_simple_proj_wing_area()))
print("Stabiliser trapz wing area =", str(stabiliser.calc_trapz_proj_wing_area()))
print("Stabiliser aspect ratio =", str(stabiliser.calc_AR()))
print("Stabiliser lift =", str(stabiliser.calc_lift(V=5, aoa=4, rho=1025)), "Newtons")

# Instantiate a mast
mast = LiftingSurface(rt_chord=150,
                      tip_chord=150,
                      span=850,
                      type='mast')
mast.generate_coords(npts=101)
# mast.plot2D()

xnode1 = np.concatenate([obj.node1.reshape(1, 1, -1) for obj in front_wing.BVs], axis=1)
xnode2 = np.concatenate([obj.node2.reshape(1, 1, -1) for obj in front_wing.BVs], axis=1)
gamma = np.array([obj.circ for obj in front_wing.BVs]).reshape(1, -1, 1)
l0 = np.array([obj.length0 for obj in front_wing.BVs]).reshape(1, -1, 1)

xcp = front_wing.xcp

V = eval_biot_savart(xcp, xnode1,  xnode2, gamma, l0)


# to-do:
# - in LiftingSurface class: designate whether a BV is on the LL or not (vtype)
# - write out circulation residual to be solved
# - code up residual, implement some auto-diff of it

