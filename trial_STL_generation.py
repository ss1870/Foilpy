#%%
import numpy as np
from scipy.interpolate import interp1d, pchip_interpolate
import AXIS_wings 
import matplotlib.pyplot as plt
from stl import mesh
%matplotlib widget

# script to trial generation of stl from lifting surface class

# instantiate a stabiliser lifting surface
re = 5 * 0.2 * 1025 / 0.00126
stab = AXIS_wings.Stab_FR_440(re, nsegs=40, plot_flag=False)

# define number of cross-sections and points around the aerofoil for the stl
# need lots for smooth geometry, particularly near tip and at LE
ncs = 100
ncs_pts = 200

# prep and interpolate airfoil on new grid
# smoothing (pchip) interpolation of afoil on evenly spaced arc-length grid
afoil_coords_in, indx = np.unique(stab.afoil_coords, return_index=True, axis=0)
afoil_coords_in = stab.afoil_coords[np.sort(indx),:]
s_coord = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(afoil_coords_in, axis=0) ** 2, axis=1))))
new_s_grid = np.linspace(s_coord[0], s_coord[-1], ncs_pts)
afoil_coords = pchip_interpolate(s_coord, afoil_coords_in, new_s_grid)

fig, ax = plt.subplots()
ax.plot(afoil_coords_in[:,0], afoil_coords_in[:,1], 'r-')
ax.plot(afoil_coords[:,0], afoil_coords[:,1], 'b-')
ax.axis('scaled')
ax.grid(True)
plt.show()

# TE points that are close are set to TE
TE_mask = np.all(np.isclose(afoil_coords, [1,0]), axis=1)
afoil_coords[TE_mask,:] = [1,0]

# move airfoil centre to (0,0)
afoil_coords[:,0] = afoil_coords[:,0] - 0.5

# add airfoil x coordinates as zero
afoil_coords = np.hstack((np.zeros((ncs_pts,1)), afoil_coords))


# interpolate LE, TE, ref_axis, washout on new spanwise grid
x_interp = np.linspace(stab.x[0], stab.x[-1], ncs)
LEf = interp1d(stab.x, stab.LE, axis=0)
LE = LEf(x_interp)
TEf = interp1d(stab.x, stab.TE, axis=0)
TE = TEf(x_interp)
ref_axisf = interp1d(stab.x, stab.ref_axis, axis=0)
ref_axis = ref_axisf(x_interp)
washoutf = interp1d(stab.x, stab.washout_curve, axis=0)
washout = washoutf(x_interp)
chord = np.linalg.norm(LE - TE, axis=1) # compute chord


# preallocate
vertices = np.empty((0,3))
faces = np.empty((0,3))


# define ID table for grid of face nodes
ID_table = np.arange(1, (ncs_pts-1)*(ncs-2)+1).reshape(-1,(ncs_pts-1)).T
ID_table = np.vstack((ID_table, ID_table[0,:]))
ID_table = np.hstack((np.zeros((ncs_pts,1)), 
                      ID_table, 
                      (np.max(ID_table)+1)*np.ones((ncs_pts,1))))

# loop through cross-sections
for i in range(ncs-1):

    # scale afoil coords (afoil is centred on (y,z)=(0,0))
    coords1 = afoil_coords * chord[i] # afoil at station 1
    coords2 = afoil_coords * chord[i+1] # afoil at station 2

    # rotate normalised coordinates by washout (rotates about (0,0))
    c = np.cos(washout[i:i+2] * np.pi/180)
    s = np.sin(washout[i:i+2] * np.pi/180)
    coords1[:,1] = coords1[:,1] * c[0] - coords1[:,2] * s[0]
    coords1[:,2] = coords1[:,1] * s[0] + coords1[:,2] * c[0]
    coords2[:,1] = coords2[:,1] * c[1] - coords2[:,2] * s[1]
    coords2[:,2] = coords2[:,1] * s[1] + coords2[:,2] * c[1]

    # rotate by anhedral?

    # shift normalised coordinates onto reference axis
    coords1 = ref_axis[i,:] - coords1
    coords2 = ref_axis[i+1,:] - coords2

    # add vertices and faces
    if i == 0 and chord[0] == 0:
        # start: single point to cross-section
        vertices = np.vstack((vertices, np.vstack((coords1[0,:], coords2[:-1,:]))))
        faces = np.vstack((faces, 
                           np.stack((ID_table[:-1,0], ID_table[:-1,1], ID_table[1:,1]), axis=1)))

    elif i == ncs - 2 and chord[i+1] == 0:
        # end: cross-section to single point
        vertices = np.vstack((vertices, coords2[0,:]))
        faces = np.vstack((faces, np.stack((ID_table[:-1,i], ID_table[:-1,i+1], ID_table[1:,i]), axis=1)))

    elif np.any(chord[i:i+2] != 0):
        # middle: cross-section to cross-section
        vertices = np.vstack((vertices, coords2[:-1,:]))
        faces1 = np.stack((ID_table[:-1,i], ID_table[:-1,i+1], ID_table[1:,i+1]), axis=1)
        faces2 = np.stack((ID_table[:-1,i], ID_table[1:,i+1], ID_table[1:,i]), axis=1)
        faces = np.vstack((faces, faces1, faces2))


# plot triangle surface
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_trisurf(vertices[:,0], vertices[:,1], faces, vertices[:,2])
ax.set_xlim3d(-stab.span*1.1/2, stab.span*1.1/2)
ax.set_ylim3d(-stab.span*1.1/2, stab.span*1.1/2)
ax.set_zlim3d(-stab.span*1.1/2, stab.span*1.1/2)

# generate stl mesh
wing = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        wing.vectors[i][j] = vertices[int(f[j]),:]

# Write the mesh to file "cube.stl"
wing.save('wing.stl')