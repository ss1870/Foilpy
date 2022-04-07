
#%%

from source.classes import LiftingSurface, FoilAssembly, ms2knts, knts2ms
from source.LL_functions import steady_LL_solve
import numpy as np
import AXIS_wings 
# %matplotlib widget

u = 5  # flow speed in m/s
chord = 0.2  # characteristic length
rho = 1025
re = u * chord * rho / 0.00126
print("Reynolds number = ", str(re), "\n")

# Define front wing
front_wing = AXIS_wings.BSC_810(re, afoil='naca1710', nsegs=40, plot_flag=True)

# Define stabiliser
stab = AXIS_wings.Stab_FR_440(re, nsegs=40, plot_flag=False)

# Define mast
mast = LiftingSurface(rt_chord=130,
                      tip_chord=130,
                      span=750,
                      Re=re,
                      type='mast',
                      afoil='naca0015',
                      nsegs=4,
                      units='mm',
                      plot_flag=False) 

# Assemble foil
foil = FoilAssembly(front_wing,
                    stab,
                    mast,
                    fuselage_length=699 - 45 - 45,  # assumes AXIS short black fuselage
                    mast_attachment_ratio=267 - 45,  # assumes AXIS short black fuselage
                    wing_angle=1,
                    stabiliser_angle=-2,
                    units='mm')

# lifting_surfaces = foil.surface2dict()
# u_motion = np.array([[0, knts2ms(15), 0]])
# out = steady_LL_solve(lifting_surfaces, -u_motion, rho, dt=0.05, nit=30, reflected_wake=True, water_surface=-0.5, wake_rollup=False, variable_time_step=False)
# print(foil.compute_foil_loads(-u_motion, rho, out[0]))

# wake_elmt_table = out[3]
# elmtIDs = out[5]

# foil.plot_wake(lifting_surfaces, wake_elmt_table, elmtIDs)

angle = np.linspace(-5,10,4)
u_motion = np.array([[0, knts2ms(7), 0],
                     [0, knts2ms(10), 0],
                     [0, knts2ms(15), 0]])
foil.analyse_foil(angle, -u_motion, rho, reflected_wake=False, compare_roll_up=False)