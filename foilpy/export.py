from socket import socket
import numpy as np
from foilpy.utils import cosspace, rotation_matrix, apply_rotation, unique_unsrt, reinterp_arc_length, sinspace
from scipy.interpolate import interp1d, CubicSpline, splprep, splev, pchip_interpolate, PchipInterpolator
import matplotlib.pyplot as plt
from copy import deepcopy
import foilpy.splines.curve as spl

def gen_single_section(coord_interper, ncs_pts, t2c, chord, washout, 
                    ref_axis, mounting_angle, TE_thick, plot_flag=False):
    """
    Generate precise coordinates for a single afoil section.
    1. Interpolates coordinates based on t2c
    2. Scales coordinates by local chord length
    3. Trims trailing edge to requested thickness
    4. Rotates coordinates by washout (i.e. about local origin)
    5. Shifts coordinates onto global reference axis
    6. Rotates coordinates by mounting angle (ie. about global origin)
    """
    ## Scale afoil coords (afoil is centred on (y,z)=(0,0))
    coords = coord_interper(t2c)
    coords = coords * chord # afoil at station 1

    ## Cut-off TE for a given min TE thickness
    # interpolate coords over chord-wise grid
    LE_ID = np.argmax(coords[:,1]) # get ID of LE (max y location)
    SS_coords = unique_unsrt(coords[:LE_ID+1,:], axis=0)
    PS_coords = np.flipud(unique_unsrt(coords[LE_ID:,:], axis=0))
    y_fine = np.linspace(coords[np.argmin(coords[:,1]), 1], coords[LE_ID,1], 1000)
    SS_coords_fine = pchip_interpolate(SS_coords[:,1], SS_coords[:,2], y_fine)
    PS_coords_fine = pchip_interpolate(PS_coords[:,1], PS_coords[:,2], y_fine)
    # compute height and find cut-off y location
    height = SS_coords_fine - PS_coords_fine
    ID = np.where(height > TE_thick/1e3)[0][0] + 10
    y_cutoff = pchip_interpolate(height[:ID], y_fine[:ID], TE_thick/1e3)
    # y_cutoff = coords[0,1] + y_cutoff
    # find z locations on SS and PS
    SS_cutoff_z = pchip_interpolate(SS_coords[:,1], SS_coords[:,2], y_cutoff)
    PS_cutoff_z = pchip_interpolate(PS_coords[:,1], PS_coords[:,2], y_cutoff)
    SS_coords = SS_coords[SS_coords[:,1] > y_cutoff, :]
    SS_coords = np.insert(SS_coords, 0, [0, y_cutoff, SS_cutoff_z], axis=0)
    PS_coords = PS_coords[PS_coords[:,1] > y_cutoff, :]
    PS_coords = np.insert(PS_coords, 0, [0, y_cutoff, PS_cutoff_z], axis=0)
    coords = unique_unsrt(np.concatenate((SS_coords, np.flipud(PS_coords)), axis=0), axis=0)
    le_id = np.argmax(coords[:,1])
    coords = reinterp_arc_length(coords, ncs_pts, keepLE=True, le_id=le_id)
    le_id = np.argmax(coords[:,1])

    ## Rotate normalised coordinates by washout (rotates about local (0,0))
    R = rotation_matrix([1,0,0], washout)
    coords = apply_rotation(R, coords, dim=1)

    # rotate by anhedral?

    ## Shift normalised coordinates onto reference axis
    coords = ref_axis + coords

    ## Rotate by mounting angle now in global CSYS
    if mounting_angle != 0:
        R = rotation_matrix([1,0,0], mounting_angle)
        coords = apply_rotation(R, coords, dim=1)

    if plot_flag:        
        fig, ax = plt.subplots()
        ax.plot(coords[:,1], coords[:,2], marker='.')
        ax.axis('scaled')
        ax.grid(True)

    return coords, le_id

def prep_afoils(afoil_table, ncs_pts):
    """
    Loops through set of normalised afoils and pre-processes them.
    Interpolate on constant equidistance arc-length distribution.
    Ensure LE and TE are present.
    Return an afoil interpolator object based on thickness-to-chord.
    """
    if (ncs_pts % 2) == 0:
        raise Exception("No of cs pts for raw afoils should be negative.")

    afoil_coords = np.zeros((ncs_pts, 3, len(afoil_table)))
    # loop through and pre-process afoils
    for i, afoil in enumerate(afoil_table):
        # prep and interpolate airfoil on new grid
        # smoothing (pchip) interpolation of afoil on evenly spaced arc-length grid
        # must ensure LE ID falls on same index of array for each afoil
        afoil_coords_in = unique_unsrt(afoil_table[afoil]["coords"], axis=0)
        le_id = np.where(np.all(np.isclose(afoil_coords_in, 0, rtol=1e-6), axis=1))[0][0]
        SS = reinterp_arc_length(afoil_coords_in[:le_id+1,:], np.ceil(ncs_pts/2).astype(int))
        SS[np.isclose(SS,[0,0])] = 0
        PS = reinterp_arc_length(afoil_coords_in[le_id:,:], np.ceil(ncs_pts/2).astype(int))
        PS[np.isclose(PS,[0,0])] = 0
        afoil_coords[:,1:,i] = np.vstack((SS, PS[1:,:]))

        # fig, ax = plt.subplots()
        # ax.plot(afoil_coords_in[:,0], afoil_coords_in[:,1])
        # ax.plot(afoil_coords[:,1,i], afoil_coords[:,2,i])

        # Points that are close to TE are set to TE
        TE_mask = np.all(np.isclose(afoil_coords[:,:,i], [0,1,0]), axis=1)
        afoil_coords[TE_mask,:,i] = [0,1,0]
        # move airfoil centre to (0,0)
        afoil_coords[:,1,i] = afoil_coords[:,1,i] - 0.5
        # flip y axis so LE is +ve 0.5 in y
        afoil_coords[:,1,i] = - afoil_coords[:,1,i]

    # create afoil interpolator
    if len(afoil_table) > 1:
        rel_thick = np.array([afoil_table[afoil]['rel_thick'] for afoil in afoil_table])
        id_sort = np.argsort(rel_thick)
        afoil_coords_interpolator = PchipInterpolator(rel_thick[id_sort],
                                    afoil_coords[:,:,id_sort], axis=2)
        
    else:
        afoil_coords_interpolator = lambda x: np.squeeze(afoil_coords)

    return afoil_coords, afoil_coords_interpolator

def create_socket_coords(origin=[0,0], radius=False, plot_flag=False):
    """
    Create coordinates of AXIS socket shape
    """
    pt = [0,0]
    # add first arc
    arc_cntr = [10.197, -384.865]
    arc_radius = 385
    strt_angle = -1.52 # deg
    end_angle = +15.29 # deg
    theta = np.linspace(strt_angle, end_angle, 100)
    x = arc_cntr[0] + arc_radius * np.sin(theta*np.pi/180)
    y = arc_cntr[1] + arc_radius * np.cos(theta*np.pi/180)

    # add linear slope
    x = np.append(x, x[-1] + 13)
    y = np.append(y, y[-1] - 13 * np.tan(15*np.pi/180))

    # add vertical line
    x = np.append(x, x[-1])
    y = np.append(y, y[-2] - 6.4)

    # add radius to back corner
    if radius:
        arc_radius1 = 1.5
        centre = np.array([x[-1]-arc_radius1, y[-2]-1.151])
        theta1 = np.linspace(75, 0, 15)
        x = np.delete(x, -2)
        y = np.delete(y, -2)
        x = np.insert(x, -1, centre[0] + arc_radius1 * np.cos(theta1*np.pi/180))
        y = np.insert(y, -1, centre[1] + arc_radius1 * np.sin(theta1*np.pi/180))
        

    coords = np.stack((x,y), axis=1)
    coords += origin
    coords /= 1e3 # convert to meters
    coords[:,0] = -coords[:,0] # reverse x axis
    coords = np.flipud(coords)
    
    if plot_flag:
        fig, ax = plt.subplots()
        ax.plot(x, y) 
        ax.axis('scaled')
        ax.grid(True)

    return coords

def get_pt_spacing(wing, resolution):
    """
    Define spanwise and cross-section no of points based on resolution request.
    """
    ## Determine point spacing based on resolution request
    # compute arc length around root chord
    afoil_coords_cntr = wing.afoil_table[wing.afoil[0][0]]["coords"]
    _, indx = np.unique(afoil_coords_cntr, return_index=True, axis=0)
    afoil_coords_cntr = afoil_coords_cntr[np.sort(indx),:] * wing.rt_chord
    s_coord_root = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(afoil_coords_cntr, axis=0) ** 2, axis=1))))
    arc_length_root = s_coord_root[-1]

    # Define spacing based on resolution
    if resolution == 'high':
        spanwise_spacing = 0.0022   # 201 pts on stab440
        cs_spacing = 0.00073        # 251 cs pts on stab440
    elif resolution == 'medium':
        spanwise_spacing = 0.0044   # 101 pts on stab440
        cs_spacing = 0.0012         # 151 cs pts on stab440
    elif resolution == 'low':
        spanwise_spacing = 0.021    # 21 pts on stab440
        cs_spacing = 0.0073         # 25 cs pts on stab440

    # Compute spanwise and arc-length-wise no of points
    ncs_pts = int(arc_length_root / cs_spacing)
    ncs = int(wing.span / spanwise_spacing)
    return ncs, ncs_pts

def spanwise_geom(wing, ncs, tip_thick, span_spacing='cosspace',
            add_socket=False, half=False, x_tuck_te=0.016):
    """
    Create spanwise geometry inputs i.e. chord, t2c, washout, ref axis
    """
    ## Create geometry interpolators
    LEf         = interp1d(wing.x, wing.LE, axis=0)
    TEf         = interp1d(wing.x, wing.TE, axis=0)
    ref_axisf   = interp1d(wing.x, wing.ref_axis, axis=0)
    t2cf        = interp1d(wing.x, wing.t2c_distribution, axis=0)
    washoutf    = interp1d(wing.x, wing.washout_curve, axis=0)

    ## Create new spanwise (x) spacing with requested ncs
    x0 = wing.x[0]
    if half:
        x0 = 0
    if span_spacing == 'cosspace':
        x_interp = cosspace(x0, wing.x[-1], n_pts=ncs)
    elif span_spacing == 'linspace':
        x_interp = np.linspace(x0, wing.x[-1], ncs)

    ## Interpolate LE, TE, ref_axis, t2c, washout on new spanwise grid
    ref_axis    = ref_axisf(x_interp)
    t2c         = t2cf(x_interp)
    washout     = washoutf(x_interp)
    LE          = LEf(x_interp)
    TE          = TEf(x_interp)
    chord       = np.linalg.norm(LE - TE, axis=1) # compute chord

    if tip_thick > 0:
        ## Trim tips given a min absolute thickness
        abs_thick = chord * t2c # Get absolute thickness
        if half:
            x_cut_off = pchip_interpolate(np.flip(abs_thick), np.flip(x_interp), tip_thick/1e3)
            x_cut_off0 = 0
        else:
            ID = np.where(abs_thick > tip_thick/1e3)[0][0] + 10
            # Find x location to cut off at
            x_cut_off = pchip_interpolate(abs_thick[:ID], x_interp[:ID], tip_thick/1e3)
            x_cut_off0 = deepcopy(x_cut_off)
        # Create new shorter xgrid
        if span_spacing == 'cosspace':
            x_interp = cosspace(x_cut_off0, np.abs(x_cut_off), n_pts=ncs)
        elif span_spacing == 'linspace':
            x_interp = np.linspace(x_cut_off0, np.abs(x_cut_off), ncs)
        # Reinterpolate geometrical parameters on new shorter xgrid
        ref_axis = ref_axisf(x_interp)
        t2c = t2cf(x_interp)
        washout = washoutf(x_interp)
        LE = LEf(x_interp)
        TE = TEf(x_interp)
        chord = np.linalg.norm(LE - TE, axis=1) # compute chord
    method = np.zeros((ncs, 1))

    if add_socket:
        # Define keypoints with fixed types at desired x locations
        KPs = np.array([
                        [0, 1],
                        [0.0145, 2],    # 29 mm
                        [x_tuck_te, 4],
                        [0.028, 3],     # 56 mm
                        [0.04, 0.5]])
        if not half:
            KPlhs = deepcopy(KPs)
            KPlhs[:,0] = -KPlhs[:,0]
            KPs = np.unique(np.vstack((np.flipud(KPlhs), KPs)), axis=0)

        # Add KPs into rowise sorted [x, method] grid
        x = np.hstack((x_interp.reshape(-1,1), method))
        x = np.vstack((x, KPs))
        x = x[x[:, 0].argsort()]
        x = x[np.any(x!=[0,0], axis=1),:]

        # Set method = 2
        x2 = x[x[:,1] == 2,0]
        if half:
            mask1 = (x[:,0] < x2[0]) & (x[:,1] != 1)
        else:
            mask1 = (x[:,0] > x2[0]) & (x[:,0] < x2[1]) & (x[:,1] != 1)
        x[mask1,1] = 2

        # between 2/3
        x3 = x[x[:,1] == 3,0]
        if half:
            mask2 = ((x[:,0] > x2[0]) & (x[:,0] < x3[0]))
        else:
            mask2 = ((x[:,0] > x3[0]) & (x[:,0] < x2[0])) | ((x[:,0] > x2[1]) & (x[:,0] < x3[1]))
        mask2 = mask2 & (x[:,1] != 4)
        # x = x[mask2==False,:]
        x[mask2,1] = 2.5

        # between 3/0.5
        x4 = x[x[:,1] == 0.5,0]
        if half:
            mask3 = ((x[:,0] > x3[0]) & (x[:,0] < x4[0]))
        else:
            mask3 = ((x[:,0] > x4[0]) & (x[:,0] < x3[0])) | ((x[:,0] > x3[1]) & (x[:,0] < x4[1]))
        # x = x[mask3==False,:]
        x[mask3,1] = 2.5
        x[x[:,1] == 0.5,1] = 0
        # Decide what to do inbetween KPs - pchip interpolate?


        # x = np.array([
        #             #   [-0.057/2, 2],
        #             #   [-0.055/2, 2],
        #             #   [-0.029/2, 2],
        #             #   [-0.027/2, 2],
        #               [0, 1],
        #               [0.027/2, 2],
        #               [0.029/2, 2],
        #             #   [0.055/2, 3],
        #               [0.056/2, 3],
        #             #   [0.031, 0],
        #             #   [0.034, 0],
        #             #   [0.040, 0],
        #               [0.045, 0],
        #               [0.06, 0]])
        # x1 = sinspace(0.075, np.abs(x_cut_off), 
        #                 int(np.floor(ncs/2)-x.shape[0]+1))
        # x = np.append(x, 
        #         np.stack((x1, np.zeros(len(x1))), axis=1), axis=0)
        # xlhs = deepcopy(x)
        # xlhs[:,0] = -xlhs[:,0]
        # x = np.unique(np.vstack((np.flipud(xlhs), x)), axis=0)

        method = x[:,1]
        x = x[:,0]
        
        # x = np.array([0, 0.029/2, 0.055/2, 0.057/2])
        # x = np.append(x, sinspace(0.057/2, np.abs(x_cut_off), int(np.floor(ncs/2)-2)))
        # x = np.unique(np.stack((np.flip(-x), x)))
        # method[int(np.floor(ncs/2))-3:int(np.floor(ncs/2))+4] = 2
        # method[int(np.floor(ncs/2))] = 1

        # Re-interp geometry on new spanwise spacing
        ref_axis    = ref_axisf(x)
        t2c         = t2cf(x)
        washout     = washoutf(x)
        LE          = LEf(x)
        TE          = TEf(x)
        chord       = np.linalg.norm(LE - TE, axis=1) # compute chord

    return x, ref_axis, chord, t2c, washout, method

def add_socket_2_coords(coords, le_id, SS_attach_id, ncs_pts,
            TE_thick, method=0, datum=None, add_radius=True,
            s_interp=None, plot_flag=False):
    """
    Take afoil coordinates and mdify to include the AXIS socket
    """
    ## Create and locate socket coords
    socket_xy = create_socket_coords(radius=add_radius)
    if method == 1:
        # method==1 : centre afoil, use attachment point to locate socket coords
        socket_xy += coords[SS_attach_id, 1:]
        datum = socket_xy[0,:]
    elif method == 2 or method == 3 or method == 4:
        # method==2 : outboard of centre afoil, use datum to
        # locate TE PS corner of socket coords
        socket_xy = socket_xy + (datum - socket_xy[0,:])
        # Create LE/SS blend between socket and coords
        # find closest point on coords to socket end
        id_sock_ss = np.argmin(np.linalg.norm(coords[round(0.25*coords.shape[0]):round(0.55*coords.shape[0]),1:] - socket_xy[-1,:], axis=1)) + round(0.25*coords.shape[0])
        # find point about 20 mm and 30 mm away to blend too
        id_sock_ss_p20 = id_sock_ss + (np.abs(np.linalg.norm(coords[id_sock_ss:round(0.55*coords.shape[0]),1:] - coords[id_sock_ss,1:], axis=1) - 0.02)).argmin()
        id_sock_ss_p30 = id_sock_ss_p20 + np.argmin(np.abs(np.linalg.norm(coords[id_sock_ss_p20:round(0.65*coords.shape[0]),1:] - coords[id_sock_ss_p20,1:], axis=1) - 0.01))
        # Create SS/LE blend between socket and coords
        pts = np.vstack((socket_xy[-20:,:],
                            coords[id_sock_ss_p20:id_sock_ss_p30,1:]))
        interper = CubicSpline(pts[:,0], pts)
        SSLE_blend = interper(np.linspace(pts[0,0], pts[-1,0], 30))
        id_le_ss1 = int(np.where(np.all(np.isclose(socket_xy, SSLE_blend[0,:]), axis=1))[0])
        id_le_ss2 = int(np.where(np.all(np.isclose(coords[:,1:], SSLE_blend[-1,:]), axis=1))[0])

    if method==4:
        ## Cut off Socket coords at y_tuck, thin TE, and blend
        # PSTE into real coords

        # Cut off end of socket and move new TE in
        y_tuck = 0.003
        TEnewSS = pchip_interpolate(socket_xy[1:,0], socket_xy[1:,:], (socket_xy[0,0] + y_tuck), axis=0)
        socket_xy = np.append(TEnewSS.reshape(-1,2), 
                    socket_xy[socket_xy[:,0] > TEnewSS[0],:], axis=0)
        
        # blend real ss into new TE location
        if add_radius:
            TEnewPS = TEnewSS - np.array([0, TE_thick/1e3])
        else:
            TEnewPS = TEnewSS - np.array([0, 1/1e3])
        socket_xy = np.append(TEnewPS.reshape(1,2), socket_xy, axis=0)
        id_le_ss1 = int(np.where(np.all(np.isclose(socket_xy, SSLE_blend[0,:]), axis=1))[0])

    if method == 1 or method == 2 or method == 4:
        ## Create pressure-side (lower) connection to afoil profile
        # find point closest coordinate on same Y plane
        id_ps_te1 = np.argmin(np.abs(coords[round(0.75*coords.shape[0]):,2] - socket_xy[0,1])) + round(0.75*coords.shape[0])
        # and point around 10mm back toward the LE
        id_ps_te1m10 = (np.argmin(np.abs(np.linalg.norm(coords[id_ps_te1,:] - coords[round(0.65*coords.shape[0]):id_ps_te1,:], axis=1) - 0.01)) + round(0.65*coords.shape[0]))
        # generate a few intermediate points along vector
        vector = coords[id_ps_te1,1:] - socket_xy[0,:]
        pts = socket_xy[0,:].reshape(1,2) + [[0,0], [0.6,0.6],[0.75,0.75],[0.8,0.8]] * vector.reshape(1,2)
        pts = np.append(pts, np.flipud(coords[id_ps_te1m10:id_ps_te1+1, 1:]), axis=0)
        interper = CubicSpline(pts[:,0], pts)
        PSxy = interper(np.linspace(pts[0,0], pts[-1,0], 30))
    elif method == 3:
        socket_xy = create_socket_coords(radius=add_radius)
        socket_xy = socket_xy + (datum - socket_xy[0,:])
        if True:
            ## Smooth socket coords down to real TE on SS, real PS
            # find point on socket coords 10mm right from TESS
            dist = np.linalg.norm(socket_xy - coords[0,1:], axis=1)
            id_20 = np.argmin(np.abs(dist - 0.02))
            # find point on coords 10mm from TE on SS
            id_ss_te_p10 = np.argmin(np.abs(np.linalg.norm(coords[:round(0.4*coords.shape[0]),:] - coords[0,:], axis=1) - 0.01))
            # define pts to interpolate
            pts = np.vstack((coords[:id_ss_te_p10,1:], socket_xy[id_20:id_le_ss1,:]))
            interper = CubicSpline(pts[:,0], pts)
            SSTE_blend = interper(np.linspace(pts[0,0], pts[-1,0], 80))
            # Assemble new coords
            # id_le_ss2 = SSTE_blend == socket_xy[id_le_ss1,:]
            new_coords = np.vstack((SSTE_blend, SSLE_blend,
                                    coords[id_le_ss2+1::, 1:]))
        else:
            ## Cut off end of socket coords and merge into coords
            # find socket coordinate on same y plane as SS TE coordinate
            # find point on socket line coincident in Y with the TESS
            TE_ss = np.array([np.interp(coords[0,2], socket_xy[:,1], socket_xy[:,0]), coords[0,2]])
            TE_ps = TE_ss - [0,TE_thick/1e3]
            # find point about 20 mm away from TE to blend too
            id_20TE = (np.abs(np.linalg.norm(coords[round(0.5*ncs_pts):,1:] - TE_ps, axis=1) - 0.02)).argmin() + round(0.5*ncs_pts)
            # Create SS/LE blend between socket and coords
            pts = np.vstack(([TE_ps], np.flipud(coords[(id_20TE-10):id_20TE,1:])))
            interper = CubicSpline(pts[:,0], pts)
            PSTE_blend = interper(np.linspace(pts[0,0], pts[-1,0], 30))
            ## Assemble new coordinates
            id1 = int(np.where(socket_xy[:,1] < TE_ss[1])[0][-1])
            new_coords = np.vstack(([TE_ss],
                                    socket_xy[id1+1:id_le_ss1,:],
                                    SSLE_blend,
                                    coords[id_le_ss2+1:id_20TE-10, 1:],
                                    np.flipud(PSTE_blend)))


    ## Assemble new upper, LE, and new lower into a set of coords
    if add_radius and method != 4:
        # Remove TE_thick from end of socket coords
        socket_xy = np.append((socket_xy[0,:] + [0, TE_thick/1e3]).reshape(1,2),
                        socket_xy[1:,:], axis=0)
    else:
        # Remove only bottom corner point from end of socket coords
        socket_xy = socket_xy[1:,:]
    if method == 1:
        new_coords = np.vstack((socket_xy,
                                coords[SS_attach_id+1:id_ps_te1m10, 1:],
                                np.flipud(PSxy)))
    elif method == 2 or method == 4:
        new_coords = np.vstack((socket_xy[:id_le_ss1-1,:],
                                SSLE_blend,
                                coords[id_le_ss2+1:id_ps_te1m10, 1:],
                                np.flipud(PSxy)))

    ## Re-interp coords over correct number of cs pts
    # find ID of LE pt
    le_id1 = np.where(np.all(np.isin(new_coords, coords[le_id,:]), axis=1))
    s = np.append(0, np.cumsum(np.linalg.norm(np.diff(new_coords, axis=0), axis=1)))
    s = s/s[-1]
    if np.all(s_interp == None):
        # Determine appropriate s_spacing from new_coords
        # Use adaptive knot spacing algorithm
        # (bunches knots in areas of higher geometric feaure variation)
        temp = reinterp_arc_length(new_coords, 1001,
                                    keepLE=True, le_id=int(le_id1[0]))
        s_temp = np.append(0, np.cumsum(np.linalg.norm(np.diff(temp, axis=0), axis=1)))    
        s_temp = s_temp/s_temp[-1]
        # u_bar = spl.parameterise_curve(new_coords, method='centripetal',
        #                                 plot_flag=plot_flag)
        U = spl.distribute_knots(s_temp, 3, ncs_pts+6, Q=temp,
                        method='adaptive', plot_flag=plot_flag)
        s_interp = U[3:-3]

    ## Re-interp on new s_spacing
    # Find s coodinate closest to LE
    id_s_le = np.argmin(np.abs(s_interp - s[le_id1]))
    s_interp[id_s_le] = s[le_id1]
    # re-interp
    # new_coords = pchip_interpolate(s, new_coords, s_interp, axis=0)

    new_coords = reinterp_arc_length(new_coords, ncs_pts,
                                    keepLE=True, le_id=int(le_id1[0]))

    # Re-add x dimension to new_coords
    x_coords = coords[0,0]*np.ones((ncs_pts,1))
    # x_coords = coords[:,0].reshape(-1,1)
    new_coords = np.hstack((x_coords, new_coords))
    le_id1 = np.where(np.all(np.isclose(new_coords, coords[le_id,:]), axis=1))[0][0]
    
    # Check LE is present in new_coords
    assert np.any(np.all(np.isclose(new_coords, coords[le_id,:]), axis=1))

    if plot_flag:
        fig, ax = plt.subplots()
        ax.plot(coords[:,1], coords[:,2])
        ax.plot(socket_xy[:,0], socket_xy[:,1])
        if method == 1 or method == 2:
            ax.plot(PSxy[:,0], PSxy[:,1])
        ax.scatter(new_coords[:,1], new_coords[:,2])
        # ax.plot(new_coords[:,1], new_coords[:,2], marker='.')
        ax.axis('scaled')
        ax.grid(True)

    return new_coords, datum, s_interp, le_id1

def renorm_coords(coords, mounting_angle, ref_axis, washout, chord, plot_flag=False):
    """
    Take an afoil section in absolute coordinates and renormalise.
    """
    # Undo mounting angle rotation
    R = rotation_matrix([1,0,0], -mounting_angle)
    coords = apply_rotation(R, coords, dim=1)
    # Undo ref axis shift
    coords = coords - ref_axis
    # Undo washout rotation
    R = rotation_matrix([1,0,0], -washout)
    coords = apply_rotation(R, coords, dim=1)
    # divide by chord
    coords /= chord
    if plot_flag:
        fig, ax = plt.subplots()
        ax.plot(coords[:,1], coords[:,2])
        ax.axis('scaled')
        ax.grid(True)
    return coords

    