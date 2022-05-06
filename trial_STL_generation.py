#%%
import AXIS_wings 
# %matplotlib widget

# script to trial generation of stl from lifting surface class

# instantiate a stabiliser lifting surface
re = 5 * 0.2 * 1025 / 0.00126
stab = AXIS_wings.Stab_FR_440(re, nsegs=40, plot_flag=False)

stab.export_wing_2_stl('stab_440_high_res', mounting_angle=-2, resolution='high')

# mounting_angle = -2
# res = 'high' # 'low'

# if res == 'high':
#     ncs = 201
#     ncs_pts = 251
# elif res == 'low':
#     ncs = 21
#     ncs_pts = 25  

# # prep and interpolate airfoil on new grid
# # smoothing (pchip) interpolation of afoil on evenly spaced arc-length grid
# afoil_coords_in, indx = np.unique(stab.afoil_coords, return_index=True, axis=0)
# afoil_coords_in = stab.afoil_coords[np.sort(indx),:]
# s_coord = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(afoil_coords_in, axis=0) ** 2, axis=1))))
# new_s_grid = np.linspace(s_coord[0], s_coord[-1], ncs_pts)
# # n_cs_pts_split = [int(ncs_pts/4), ncs_pts - 2*int(ncs_pts/4), int(ncs_pts/4)]
# # s_split = [s_coord[0], s_coord[-1]/4, s_coord[-1]*3/4, s_coord[-1]]
# # new_s_grid = np.concatenate((np.linspace(s_split[0], s_split[1], n_cs_pts_split[0]),
# #                              cosspace(s_split[1], s_split[2], n=-n_cs_pts_split[1], factor=0),
# #                              np.linspace(s_split[2], s_split[3], n_cs_pts_split[2])
# #                              ))
# # new_s_grid = cosspace(s_coord[0], s_coord[-1], n=-ncs_pts, factor=0.5)
# afoil_coords = pchip_interpolate(s_coord, afoil_coords_in, new_s_grid)

# fig, ax_afoil = plt.subplots()
# ax_afoil.plot(afoil_coords_in[:,0], afoil_coords_in[:,1], 'r-', label ='Input')
# ax_afoil.plot(afoil_coords[:,0], afoil_coords[:,1], 'b-', marker=None, label ='Interpolated')
# ax_afoil.legend()
# ax_afoil.axis('scaled')
# ax_afoil.grid(True)
# plt.show()

# # TE points that are close are set to TE
# TE_mask = np.all(np.isclose(afoil_coords, [1,0]), axis=1)
# afoil_coords[TE_mask,:] = [1,0]

# # move airfoil centre to (0,0)
# afoil_coords[:,0] = afoil_coords[:,0] - 0.5

# # add airfoil x coordinates as zero
# afoil_coords = np.hstack((np.zeros((ncs_pts,1)), afoil_coords))


# # interpolate LE, TE, ref_axis, washout on new spanwise grid
# # x_interp = np.linspace(stab.x[0], stab.x[-1], ncs)
# x_interp = cosspace(stab.x[0], stab.x[-1], n=ncs)
# LEf = interp1d(stab.x, stab.LE, axis=0)
# LE = LEf(x_interp)
# TEf = interp1d(stab.x, stab.TE, axis=0)
# TE = TEf(x_interp)
# ref_axisf = interp1d(stab.x, stab.ref_axis, axis=0)
# ref_axis = ref_axisf(x_interp)
# washoutf = interp1d(stab.x, stab.washout_curve, axis=0)
# washout = washoutf(x_interp)
# chord = np.linalg.norm(LE - TE, axis=1) # compute chord


# # preallocate
# vertices = np.empty((0,3))
# faces = np.empty((0,3))


# # define ID table for grid of face nodes
# ID_table = np.arange(1, (ncs_pts-1)*(ncs-2)+1).reshape(-1,(ncs_pts-1)).T
# ID_table = np.vstack((ID_table, ID_table[0,:]))
# ID_table = np.hstack((np.zeros((ncs_pts,1)), 
#                       ID_table, 
#                       (np.max(ID_table)+1)*np.ones((ncs_pts,1))))

# # loop through cross-sections
# for i in range(ncs-1):

#     # scale afoil coords (afoil is centred on (y,z)=(0,0))
#     coords1 = afoil_coords * chord[i] # afoil at station 1
#     coords2 = afoil_coords * chord[i+1] # afoil at station 2

#     # rotate normalised coordinates by washout (rotates about local (0,0))
#     c = np.cos(washout[i:i+2] * np.pi/180)
#     s = np.sin(washout[i:i+2] * np.pi/180)
#     R1 = rotation_matrix([1,0,0], washout[i])
#     R2 = rotation_matrix([1,0,0], washout[i+1])
#     coords1 = apply_rotation(R1, coords1, dim=1)
#     coords2 = apply_rotation(R2, coords2, dim=1)
#     # coords1[:,1] = coords1[:,1] * c[0] - coords1[:,2] * s[0]
#     # coords1[:,2] = coords1[:,1] * s[0] + coords1[:,2] * c[0]
#     # coords2[:,1] = coords2[:,1] * c[1] - coords2[:,2] * s[1]
#     # coords2[:,2] = coords2[:,1] * s[1] + coords2[:,2] * c[1]

#     # rotate by anhedral?

#     # shift normalised coordinates onto reference axis
#     coords1 = ref_axis[i,:] - coords1
#     coords2 = ref_axis[i+1,:] - coords2

#     # rotate by mounting angle (rotates about global (0,0))
#     if mounting_angle != 0:
#         R1 = rotation_matrix([1,0,0], mounting_angle)
#         R2 = rotation_matrix([1,0,0], mounting_angle)
#         coords1 = apply_rotation(R1, coords1, dim=1)
#         coords2 = apply_rotation(R2, coords2, dim=1)

#     # add vertices and faces
#     if i == 0 and chord[0] == 0:
#         # start: single point to cross-section
#         vertices = np.vstack((vertices, np.vstack((coords1[0,:], coords2[:-1,:]))))
#         faces = np.vstack((faces, 
#                            np.stack((ID_table[:-1,0], ID_table[:-1,1], ID_table[1:,1]), axis=1)))

#     elif i == ncs - 2 and chord[i+1] == 0:
#         # end: cross-section to single point
#         vertices = np.vstack((vertices, coords2[0,:]))
#         faces = np.vstack((faces, np.stack((ID_table[:-1,i], ID_table[:-1,i+1], ID_table[1:,i]), axis=1)))

#     elif np.any(chord[i:i+2] != 0):
#         # middle: cross-section to cross-section
#         vertices = np.vstack((vertices, coords2[:-1,:]))
#         faces1 = np.stack((ID_table[:-1,i], ID_table[:-1,i+1], ID_table[1:,i+1]), axis=1)
#         faces2 = np.stack((ID_table[:-1,i], ID_table[1:,i+1], ID_table[1:,i]), axis=1)
#         faces = np.vstack((faces, faces1, faces2))


# # plot triangle surface
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ax.plot_trisurf(vertices[:,0], vertices[:,1], faces, vertices[:,2])
# ax.set_xlim3d(-stab.span*1.1/2, stab.span*1.1/2)
# ax.set_ylim3d(-stab.span*1.1/2, stab.span*1.1/2)
# ax.set_zlim3d(-stab.span*1.1/2, stab.span*1.1/2)

# # generate stl mesh
# wing = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces):
#     for j in range(3):
#         wing.vectors[i][j] = vertices[int(f[j]),:]

# # Write the mesh to file "cube.stl"
# wing.save('wing.stl')