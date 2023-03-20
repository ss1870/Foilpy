#%%
import numpy as np
import AXIS_wing_definitions as AX_wings
from foilpy.foildef import FoilAssembly
from foilpy.muxWrapper import MachUpXWrapper
import machupX as MX
import json
# %matplotlib widget

U = 5  # flow speed in m/s
CHORD = 0.2  # characteristic length
RHO = 1025
RE = U * CHORD * RHO / 0.00126
print("Reynolds number = ", str(RE), "\n")

# Define front wing
front_wing = AX_wings.bsc_810(RE, nsegs=40, plot_flag=False)
# Define stabiliser
stab = AX_wings.stab_fr_440(RE, nsegs=40, plot_flag=False)
# Define mast
mast = AX_wings.mast_75cm(RE, nsegs=8, plot_flag=False)
# Assemble foil
foil = FoilAssembly(front_wing,
                    stab,
                    mast,
                    fuselage_length=699 - 45 - 45,  # assumes AXIS short black fuselage
                    mast_attachment_ratio=267 - 45,  # assumes AXIS short black fuselage
                    wing_angle=1,
                    stabiliser_angle=-2,
                    units='mm')

baseDir = 'C:/Git/foilpy/examples'

mux = MachUpXWrapper(baseDir)
mux.foilFromFoilAssembly(foil)
mux.setVelocitykts(15.0)
mux.setAngleDeg(2.0)

my_scene = MX.Scene(mux._inputDict)
my_scene.display_wireframe(show_legend=True)
FM_results = my_scene.solve_forces(dimensional=True, non_dimensional=False, verbose=True)
print(json.dumps(FM_results["foil1"]["total"], indent=4))
