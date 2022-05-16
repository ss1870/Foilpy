
#%%
import numpy as np
import AXIS_wing_definitions as AX_wings
from pyfoil.classes import LiftingSurface, FoilAssembly, knts2ms
from pyfoil.LL_functions import steady_LL_solve
# %matplotlib widget

U = 5  # flow speed in m/s
CHORD = 0.2  # characteristic length
RHO = 1025
RE = U * CHORD * RHO / 0.00126
print("Reynolds number = ", str(RE), "\n")

# Define front wing
front_wing = AX_wings.BSC_810(RE, afoil='naca1710', nsegs=40, plot_flag=True)

# Define stabiliser
stab = AX_wings.Stab_FR_440(RE, nsegs=40, plot_flag=False)

# Define mast
mast = LiftingSurface(rt_chord=130,
                      tip_chord=130,
                      span=750,
                      Re=RE,
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

lifting_surfaces = foil.surface2dict()
u_motion = np.array([[0, knts2ms(10), 0]])
out = steady_LL_solve(lifting_surfaces, -u_motion, RHO,
                      dt=0.05, nit=30, reflected_wake=False,
                      wake_rollup=False, variable_time_step=False)
print(foil.compute_foil_loads(-u_motion, RHO, out[0]))

wake_elmt_table = out[3]
elmtIDs = out[5]

foil.plot_wake(lifting_surfaces, wake_elmt_table, elmtIDs)

angle = np.linspace(-5,10,4)
u_motion = np.array([[0, knts2ms(7), 0],
                     [0, knts2ms(10), 0],
                     [0, knts2ms(15), 0]])
foil.analyse_foil(angle, -u_motion, RHO, reflected_wake=False, compare_roll_up=False)

stab.export_wing_2_stl('wing.stl')
