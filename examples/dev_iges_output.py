#%%
import numpy as np
import matplotlib.pyplot as plt
import AXIS_wing_definitions as AX_wings
from foilpy.utils import cosspace, rotation_matrix, apply_rotation, unique_unsrt, reinterp_arc_length
from foilpy.export import create_socket_coords, renorm_coords, gen_single_section, prep_afoils, add_socket_2_coords, spanwise_geom, get_pt_spacing
from scipy.interpolate import interp1d, CubicSpline, splprep, splev, pchip_interpolate, PchipInterpolator
%matplotlib widget

## Inputs
U = 5  # flow speed in m/s
CHORD = 0.2  # characteristic length
RHO = 1025
RE = U * CHORD * RHO / 0.00126
wing = AX_wings.bsc_810(RE, nsegs=40, plot_flag=False)

export_type = 'points' # points, stl
SF=1000
mounting_angle=1.5
plot_flag=True
resolution='low'
ncs_pts = 61
ncs = 45
tip_thick = 2 # mm 
TE_thick = 0.5 # mm

if export_type == 'stl':
    ncs, ncs_pts = get_pt_spacing(wing, resolution)

if (ncs_pts % 2) == 0:
    ncs_pts += 1 # if even, add 1
if (ncs % 2) == 0:
    ncs += 1 # if even, add 1
print('Number of cross-sectional points = ', str(ncs_pts))
print('Number of spanwise points = ', str(ncs))


## Prep afoils and generate afoil interpolator based on t2c
afoil_coords, afoil_coords_interpolator = prep_afoils(wing.afoil_table, ncs_pts)


## Create spanwise geometry inputs i.e. chord, t2c, washout, ref axis
ref_axis, chord, t2c, washout, method = spanwise_geom(wing, ncs, tip_thick, add_socket=True)



## Modify wing centre with socket shape
## Generate central afoil coords
i = np.where(np.isclose(ref_axis[:,0], 0))[0][0]
coords, le_id = gen_single_section(afoil_coords_interpolator, ncs_pts, t2c[i], 
                                chord[i], washout[i], ref_axis[i,:], mounting_angle, 
                                TE_thick, plot_flag=False)
SS_attach_id = 22
new_coords, datum = add_socket_2_coords(coords, le_id, SS_attach_id, 
                                    ncs_pts, TE_thick, method=1,
                                    plot_flag=True)


## Add socket to coords, outboard of center
# same as before, but with a blend on the SS near the LE
# ii = int(np.floor(ncs/2)) + 4
# coords1, le_id1 = gen_single_section(afoil_coords_interpolator, ncs_pts, t2c[ii], 
#                                 chord[ii], washout[ii], ref_axis[ii,:], mounting_angle, 
#                                 TE_thick, plot_flag=False)

# new_coords1, _ = add_socket_2_coords(coords1, le_id1, [], 
#                                     ncs_pts, TE_thick, method=3,
#                                     datum=datum, plot_flag=True)
# fig, ax = plt.subplots()
# ax.plot(coords[:,1], coords[:,2])
# ax.plot(socket_xy[:,0], socket_xy[:,1])
# ax.scatter(PSTE_blend[:,0], PSTE_blend[:,1])
# ax.axis('scaled')
# ax.grid(True)



# ## Re-normalise afoil
# norm_coords = renorm_coords(np.hstack((np.zeros((ncs_pts,1)), new_coords)), 
#                             mounting_angle, ref_axis[i,:], washout[i], chord[i])
# ## Insert new afoil(s) into the afoil-table, re-declare afoil interpolator
# afoilt2c = [wing.afoil_table[afoil]['rel_thick'] for afoil in wing.afoil_table]
# afoilt2c.insert(0, t2c[i]+0.01)
# afoil_coords = np.concatenate((norm_coords.reshape(-1,3,1), afoil_coords), axis=2)
# afoil_coords_interpolator = interp1d(afoilt2c, afoil_coords)
## Ensure spanwise x and t2c distribution is modified appropriately,
# in order to get correct interpolated shape
# may need to increase density of interpolation points in some areas
# ensure edges of socket are defined as fixed
# may need to include another afoil with modified top surface but normal bottom surface
# ensure TE is interpolated nicely

## To do:
# - Implement method=3 for Add socket to coords
# - work out what to do at the TE of the socket
# - Find an x spacing that minimises the spline waviness in the loft. Can I optimise this?
    # - multi-part loft might be the solution to this? if smooth enough at intersections



points_all = np.empty((0,3))
LE_pts = np.empty((0,3))
TE_pts = np.empty((0,3))
## Loop through spanwise section to generate points
for i in range(len(chord)):

    coords, le_id = gen_single_section(afoil_coords_interpolator, ncs_pts, t2c[i], 
                                    chord[i], washout[i], ref_axis[i,:], mounting_angle, 
                                    TE_thick)
    if method[i] == 1:
        coords, datum = add_socket_2_coords(coords, le_id, SS_attach_id, 
                                    ncs_pts, TE_thick, method=1,
                                    plot_flag=False)
    elif method[i] == 2 or method[i] == 3:
        coords, _ = add_socket_2_coords(coords, le_id, [], 
                                    ncs_pts, TE_thick, method=method[i],
                                    datum=datum, plot_flag=False)
    # fig, ax = plt.subplots()
    # ax.plot(coords[:,1], coords[:,2])
    # ax.axis('scaled')
    # ax.grid(True)

    # save coordinates
    points_all = np.append(points_all, coords, axis=0)
    LE_pts = np.append(LE_pts, [coords[le_id,:]], axis=0)
    TE_pts = np.append(TE_pts, [(coords[0,:]+coords[-1,:])/2], axis=0)

points_all *= SF


import csv
with open('spline_coords.txt', 'w') as f:
    csv.writer(f, delimiter=',', lineterminator='\n').writerows(points_all)

from geomdl import fitting
from geomdl.visualization import VisMPL as vis
size_u = len(chord)
size_v = ncs_pts
degree_u = 3
degree_v = 3

# Do global surface interpolation
surf = fitting.interpolate_surface(points_all, size_u, size_v, degree_u, degree_v)

# Plot the interpolated surface
surf.delta = 0.05
surf.vis = vis.VisSurface()
surf.render()
# from geomdl import exchange
# exchange.export_obj(surf, "wing_surf.obj")
# exchange.export_json(surf, "wing_surf.json")