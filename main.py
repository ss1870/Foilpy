from source.classes import LiftingSurface

front_wing = LiftingSurface(rt_chord=250,
                            tip_chord=80,
                            span=800,
                            sweep_tip=-200,
                            sweep_curv=3,
                            dih_tip=75,
                            dih_curve=15)

front_wing.generate_coords(npts=101)
front_wing.plot2D()
front_wing.plot3D()
print("Simple wing area =", str(front_wing.calc_simple_proj_wing_area()))
print("Trapz wing area =", str(front_wing.calc_trapz_proj_wing_area()))
print("Aspect ratio =", str(front_wing.calc_AR()))
print("Wing lift =", str(front_wing.calc_lift(V=5, aoa=4, rho=1025)), "Newtons")

front_wing.generate_LL_geom(10)
print("Lifting line wing area =", str(front_wing.LL_seg_area))

# print(front_wing.BVs[39].node1, front_wing.BVs[39].node2, front_wing.BVs[0].elmtID, front_wing.BVs[0].circ)
# print(len(front_wing.BVs))
# print(front_wing.BVs[-1].elmtID)
