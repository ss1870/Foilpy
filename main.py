
from source.classes import LiftingSurface

wing1 = LiftingSurface(250, 80, 800, -200, 3, 75, 15)
wing1.generate_coords(npts=101)
# wing1.plot2D()
wing1.plot3D()
print("Simple wing area =", str(wing1.calc_simple_proj_wing_area()))
print("Trapz wing area =", str(wing1.calc_trapz_proj_wing_area()))
print("Aspect ratio =", str(wing1.calc_AR()))
print("Wing lift =", str(wing1.calc_lift(V=5, aoa=4, rho=1025)), "Newtons")

# wing1.generate_LL_geom(10)
# wing1.LL_seg_area
# print(wing1.BVs[39].node1, wing1.BVs[39].node2, wing1.BVs[0].elmtID, wing1.BVs[0].circ)
# print(len(wing1.BVs))
# print(wing1.BVs[-1].elmtID)