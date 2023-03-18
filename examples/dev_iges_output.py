#%%
import numpy as np
import matplotlib.pyplot as plt
import AXIS_wing_definitions as AX_wings
from foilpy.utils import cosspace, rotation_matrix, apply_rotation, unique_unsrt, reinterp_arc_length
from foilpy.export import create_socket_coords, renorm_coords, gen_single_section, prep_afoils, add_socket_2_coords, spanwise_geom, get_pt_spacing
from scipy.interpolate import interp1d, CubicSpline, splprep, splev, pchip_interpolate, PchipInterpolator
import sys
# %matplotlib widget

import foilpy.export as ex
import importlib
import foilpy.splines.curve as spl
import foilpy.splines.surface as sur
import importlib
importlib.reload(spl)
importlib.reload(sur)

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
ncs_pts = 501   # used to interpolate modified aerofoils
ncs_pts1 = 1001 # used to interpolate raw aerofoils
ncs = 351
tip_thick = 1 # mm 
TE_thick = 0.3 # mm

if export_type == 'stl':
    ncs, ncs_pts = get_pt_spacing(wing, resolution)

if (ncs_pts % 2) == 0:
    ncs_pts += 1 # if even, add 1
if (ncs % 2) == 0:
    ncs += 1 # if even, add 1
print('Number of cross-sectional points = ', str(ncs_pts))
print('Number of spanwise points = ', str(ncs))


## Prep afoils and generate afoil interpolator based on t2c
(afoil_coords,
    afoil_coords_interpolator) = ex.prep_afoils(wing.afoil_table, 
                                                ncs_pts1)

## Create spanwise geometry inputs i.e. chord, t2c, washout, ref axis
(x, ref_axis, 
    chord, t2c, 
    washout, method) = ex.spanwise_geom(wing, ncs, tip_thick, 
                                    span_spacing='linspace', 
                                    add_socket=True, 
                                    half=True, x_tuck_te=0.017)

importlib.reload(ex)
## Modify wing centre with socket shape
## Generate central afoil coords
i = np.where(np.isclose(ref_axis[:,0], 0))[0][0]
coords, le_id = ex.gen_single_section(afoil_coords_interpolator,
                    ncs_pts1, t2c[i], chord[i], washout[i],
                    ref_axis[i,:], mounting_angle, TE_thick, 
                    plot_flag=False)
SS_attach_id = 375 #56
# (new_coords, datum, 
#     s_interp, le_id) = ex.add_socket_2_coords(coords, le_id,
#                         SS_attach_id, ncs_pts, TE_thick,
#                         method=1, plot_flag=True)
s_interp = np.linspace(0,1,ncs_pts)
(new_coords, datum, 
    _, le_id) = ex.add_socket_2_coords(coords, le_id,
                        SS_attach_id, ncs_pts, TE_thick,
                        s_interp=s_interp, method=1,
                        plot_flag=True)

## Add socket to coords, outboard of center
# same as before, but with a blend on the SS near the LE
ii = 140 #int(np.floor(ncs/2)) + 4
coords1, le_id1 = gen_single_section(afoil_coords_interpolator,
                    ncs_pts1, t2c[ii], chord[ii], washout[ii],
                    ref_axis[ii,:], mounting_angle, TE_thick,
                    plot_flag=False)

(new_coords1,
     _, _, le_id2) = ex.add_socket_2_coords(coords1, le_id1, [], 
                        ncs_pts, TE_thick, method=4,
                        s_interp=s_interp, datum=datum,
                        plot_flag=True)


# Issues:
# - How to capture detail at important points, like radius, with even u spacing along v



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


sys.exit()
points_all = np.empty((0,3))
points_all1 = np.zeros((ncs_pts,ref_axis.shape[0],3))
LE_pts = np.empty((0,3))
TE_pts = np.empty((0,3))
## Loop through spanwise section to generate points
for i in range(len(chord)):

    coords, le_id = gen_single_section(afoil_coords_interpolator,
                        ncs_pts1, t2c[i], chord[i], washout[i], 
                        ref_axis[i,:], mounting_angle, TE_thick, 
                        plot_flag=False)

    # coords = afoil_coords_interpolator(t2c[i])
    # coords = coords * chord[i]
    # fig, ax = plt.subplots()
    # ax.plot(coords[:,1], coords[:,2])
    # ax.axis('scaled')
    # ax.grid(True)

    if method[i] == 1:
        # method==1: full socket coordinates located via SS_attach_id
        coords, datum, _, le_id = add_socket_2_coords(coords, le_id, SS_attach_id, 
                                    ncs_pts, TE_thick, method=1,
                                    s_interp=s_interp, plot_flag=False)
    elif method[i] == 2 or method[i] == 3 or method[i] == 4:
        # method==2: full socket coordinates located via datum from method=1
        # method==3: only SS socket coords are added, located via datum
        coords, _, _, le_id = add_socket_2_coords(coords, le_id, [], 
                                    ncs_pts, TE_thick, method=method[i],
                                    datum=datum, s_interp=s_interp, 
                                    plot_flag=False)
    elif method[i] == 2.5:
        coords = np.zeros((ncs_pts,3))
        le_id = 0
    else:
        coords1 = reinterp_arc_length(coords, ncs_pts,
                                keepLE=True, le_id=int(le_id))
        le_id = np.where(np.all(np.isclose(coords1, coords[le_id,:]), axis=1))[0][0]
        coords = coords1
        # s = np.append(0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        # s = s/s[-1]
        # id_s_le = np.argmin(np.abs(s_interp - s[le_id]))
        # s_interp[id_s_le] = s[le_id]
        # le_id = id_s_le
        # coords = pchip_interpolate(s, coords, s_interp, axis=0)

    # save coordinates
    points_all = np.append(points_all, coords, axis=0)
    points_all1[:,i,:] = coords

    LE_pts = np.append(LE_pts, [coords[le_id,:]], axis=0)
    TE_pts = np.append(TE_pts, [(coords[0,:]+coords[-1,:])/2], axis=0)

mask = method==2.5
if np.any(mask):
    points_all1[:,mask,:] = pchip_interpolate(x[mask==False], 
                            points_all1[:,mask==False,:], 
                            x[mask], axis=1)
    LE_pts[mask,:] = pchip_interpolate(x[mask==False], 
                            LE_pts[mask==False,:], 
                            x[mask], axis=0)
    TE_pts[mask,:] = pchip_interpolate(x[mask==False], 
                            TE_pts[mask==False,:], 
                            x[mask], axis=0)

points_all *= SF
points_all1 *= SF
LE_pts *= SF
TE_pts *= SF

fig, ax = plt.subplots()
ax = fig.add_subplot(projection='3d')
for i in range(points_all1.shape[1]):
    if method[i] == 4:
        ax.plot3D(points_all1[:,i,0], points_all1[:,i,1], points_all1[:,i,2], color='red')
    else:
        ax.plot3D(points_all1[:,i,0], points_all1[:,i,1], points_all1[:,i,2], color='black')
for i in range(points_all1.shape[0]):
    ax.plot3D(points_all1[i,:,0], points_all1[i,:,1], points_all1[i,:,2], color='black')


# curve = spl.curve_approx(points_all1[:,178,:], 39, 3, plot_flag=True,
#                         knot_spacing='adaptive', param_method='Fang')
# curve.plot_curve(extra_pts=points_all1[:,178,:], scaled=True, plotCPs=True)

importlib.reload(sur)
importlib.reload(spl)
p = 3
q = 3
ncp_u = 39
ncp_v = 39

# mysurf_interp = sur.surf_interp(points_all1, p, q, param_method='chord', plot_flag=True)
mysurf_approx = sur.surf_approx(points_all1, ncp_u, ncp_v, p, q, 
                    param_method='Fang', knot_spacing='adaptive',
                    xtra_U=None, xtra_V=[0.005, 0.01, 0.015],
                    root_deriv=True,
                    int_plots=True, plot_flag=True, scatter=True)
mysurf_approx.grid_plot(npts_u=51, npts_v=301, scaled=False)

surf_name = '/mnt/c/temp/wing1'
with open(surf_name+'_U.npy', 'wb') as f:
    np.save(f, mysurf_approx.U, allow_pickle=True)
with open(surf_name+'_V.npy', 'wb') as f:
    np.save(f, mysurf_approx.V, allow_pickle=True)
with open(surf_name+'_P.npy', 'wb') as f:
    np.save(f, mysurf_approx.contrl_pts, allow_pickle=True)

# Fit LE and TE with splines
LE_spl = spl.curve_approx(LE_pts, 51, 3, plot_flag=True, 
                    knot_spacing='adaptive', param_method='Fang')
LE_spl.plot_curve(pts=100, method=2, fig = None, ax = None, 
                    return_axes=False, scaled=True, 
                    plotCPs=True)
TE_spl = spl.curve_approx(TE_pts, 51, 3, plot_flag=True, 
                    knot_spacing='adaptive', param_method='Fang')
TE_spl.plot_curve(pts=100, method=2, fig = None, ax = None, 
                    return_axes=False, scaled=True, 
                    plotCPs=True)
curve_name = '/mnt/c/temp/LE'
with open(curve_name+'_U.npy', 'wb') as f:
    np.save(f, LE_spl.U, allow_pickle=True)
with open(curve_name+'_P.npy', 'wb') as f:
    np.save(f, LE_spl.contrl_pts, allow_pickle=True)
curve_name = '/mnt/c/temp/TE'
with open(curve_name+'_U.npy', 'wb') as f:
    np.save(f, TE_spl.U, allow_pickle=True)
with open(curve_name+'_P.npy', 'wb') as f:
    np.save(f, TE_spl.contrl_pts, allow_pickle=True)

# import csv
# with open('spline_coords.txt', 'w') as f:
#     csv.writer(f, delimiter=',', lineterminator='\n').writerows(points_all)

# from geomdl import fitting
# from geomdl.visualization import VisMPL as vis
# size_u = len(chord)
# size_v = ncs_pts
# degree_u = 3
# degree_v = 3

# # Do global surface interpolation
# surf = fitting.interpolate_surface(points_all, size_u, size_v, degree_u, degree_v)

# # Plot the interpolated surface
# surf.delta = 0.05
# surf.vis = vis.VisSurface()
# surf.render()
# from geomdl import exchange
# exchange.export_obj(surf, "wing_surf.obj")
# exchange.export_json(surf, "wing_surf.json")