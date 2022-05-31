from copy import deepcopy
import os
import math
import csv
import stl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, splprep, splev, pchip_interpolate, PchipInterpolator
from foilpy.myaeropy.xfoil_module import find_coefficients
from foilpy.LL_functions import steady_LL_solve, plot_wake
from foilpy.utils import rotation_matrix, translation_matrix, apply_rotation, cosspace
from foilpy.utils import unit_2_meters, ms2knts
# import jax_cosmo as jc


class FoilAssembly:

    def __init__(self, main_wing0, stabiliser0, mast0, fuselage_length, mast_attachment_ratio,
                 wing_angle=0, stabiliser_angle=0, units='mm'):
        self.main_wing0 = main_wing0
        self.stabiliser0 = stabiliser0
        self.mast0 = mast0
        self.fuselage_length = unit_2_meters(fuselage_length, units)
        self.mast_attachment_ratio = unit_2_meters(mast_attachment_ratio, units)
        self.wing_angle = wing_angle
        self.stabiliser_angle = stabiliser_angle
        self.foil_angle = 0

        # declare current wing/mast/stab objects as deepcopies of initial wing/mast/stab objects
        self.main_wing = deepcopy(main_wing0)
        self.mast = deepcopy(mast0)
        self.stabiliser = deepcopy(stabiliser0)

        self.compute_CoG()

        # Rotate current wing/mast/stab objects into required foil arrangement
        # Assume origin (coordinate [0,0,0]) is the centre of the top of the mast
        # Mast rotate +90 degree about y axis, then translate mast.span in negative z
        R_mast = np.dot(translation_matrix([0, 0, -self.mast.span]),
                        rotation_matrix([0, 1, 0], -90))
        self.mast.rotate_component(R_mast)
        # self.mast.plot3D()
        # forces = self.mast.LL_strip_theory_forces(np.array([0,5,0]), 1025)
        # print(np.sum(forces, axis=0))

        # Front wing: rotate by wing set angle, then translate z by mast span, and y positive by attachment ratio
        R_front_wing = np.dot(translation_matrix([0, self.mast_attachment_ratio, -self.mast.span]),
                              rotation_matrix([1, 0, 0], self.wing_angle))
        self.main_wing.rotate_component(R_front_wing)
        # self.main_wing.plot3D()
        # forces = self.main_wing.LL_strip_theory_forces(np.array([0, 5, 0]), 1025)
        # print("Main wing load vector = ", str(np.sum(forces, axis=0)), "\n")

        # Stabiliser: rotate by stab set angle, then translate z by mast span, and y negative by attachment ratio
        R_stab = np.dot(translation_matrix([0, -(self.fuselage_length - self.mast_attachment_ratio), -self.mast.span]),
                        rotation_matrix([1, 0, 0], self.stabiliser_angle))
        self.stabiliser.rotate_component(R_stab)
        # self.stabiliser.plot3D()
        # forces = self.stabiliser.LL_strip_theory_forces(np.array([0, 5, 0]), 1025)
        # print("Stabiliser wing load vector = ", str(np.sum(forces, axis=0)), "\n")

    def plot_foil_assembly(self):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        fig, ax = self.main_wing.plot3D(new_fig=False, fig =fig, ax=ax)
        fig, ax = self.mast.plot3D(new_fig=False, fig=fig, ax=ax)
        fig, ax = self.stabiliser.plot3D(new_fig=False, fig=fig, ax=ax)
        ax.plot3D(self.cog[0], self.cog[1], self.cog[2], 'o')
        plt.xlabel('x')
        plt.ylabel('y - longitudinal')
        # plt.zlabel('z - upward')
        plt.show()
        return fig, ax

    def compute_CoG(self):
        # Mast
        # mast_mass = 2.1kg # 75cm axis alu mast includes baseplate, doodad and bolts (https://www.standupzone.com/forum/index.php?topic=35177.810)
        # Works out as 2.8kg per meter
        # Therefore, 60cm mast (+doodad and baseplate) would weigh 1.68kg
        self.mast0.mass = 2.8 * self.mast0.span
        self.mast0.cog = np.array([0, 0, -self.mast.span/2])

        # Front wing
        self.main_wing0.mass = 1.5 # guess
        self.main_wing0.cog = np.array([0, self.mast_attachment_ratio, -self.mast.span]) # guess

        # Stabiliser wing
        self.stabiliser0.mass = 0.4  # guess
        self.stabiliser0.cog = np.array([0, -(self.fuselage_length - self.mast_attachment_ratio), -self.mast.span])  # guess

        # Fuselage
        # fuselage_mass = 0.931 # short black
        # fuselage longitudinal CoG = 295mm from the front
        fuselage_mass = 0.931
        fuselage_cog = np.array([0, -0.030, -self.mast.span])

        # North board
        # board_mass = 3.7 # kg
        # Board CoG is 400mm infront of centre of foil mounting point. Assuming this is (0,0) on the foil CS
        # Rider will stand with backfoot roughly above foil mounting point, and front foot roughly 670mm infront of this
        board_mass = 3.7
        board_cog = np.array([0, 0.400, 0.015])

        # Me rider
        # rider_mass = 72 # kg
        # Guess rider CoG is roughly 900mm above feet and initially assume weight spread evenly between feet
        rider_mass = 75 # 72
        rider_cog = np.array([0, 0.450, 0.900]) # np.array([0, 385, 900])

        self.total_mass = self.mast0.mass + self.main_wing0.mass + self.stabiliser0.mass + fuselage_mass + board_mass + rider_mass
        cog = (self.mast0.mass*self.mast0.cog + self.main_wing0.mass*self.mast0.cog + self.stabiliser0.mass*self.mast0.cog + fuselage_mass*fuselage_cog + board_mass*board_cog + rider_mass*rider_cog)/self.total_mass
        self.cog = cog.reshape(3,1)
        print("Total mass (without rider) = ", str(self.total_mass - rider_mass), "\n")
        print("Total mass (including rider) = ", str(self.total_mass), "\n")
        print("CoG location = ", str(self.cog), "\n")

    def rotate_foil_assembly(self, rot_angle):
        # X = pitch
        # Y = roll
        # Z = yaw
        T = translation_matrix(-np.squeeze(self.cog))

        if rot_angle[2] != 0:
            rot_yaw = rotation_matrix([0, 0, 1], rot_angle[2])
            R_yaw = np.dot(np.linalg.inv(T), np.dot(rot_yaw, T))
            self.mast.rotate_component(R_yaw)
            self.main_wing.rotate_component(R_yaw)
            self.stabiliser.rotate_component(R_yaw)
            # self.cog = apply_rotation(R_yaw, self.cog, 0)

        if rot_angle[1] != 0:
            rot_roll = rotation_matrix([0, 1, 0], rot_angle[1])
            R_roll = np.dot(np.linalg.inv(T), np.dot(rot_roll, T))
            self.mast.rotate_component(R_roll)
            self.main_wing.rotate_component(R_roll)
            self.stabiliser.rotate_component(R_roll)
            # self.cog = apply_rotation(R_roll, self.cog, 0)

        if rot_angle[0] != 0:
            rot_pitch = rotation_matrix([1, 0, 0], rot_angle[0])
            R_pitch = np.dot(np.linalg.inv(T), np.dot(rot_pitch, T))
            self.mast.rotate_component(R_pitch)
            self.main_wing.rotate_component(R_pitch)
            self.stabiliser.rotate_component(R_pitch)
            # self.cog = apply_rotation(R_pitch, self.cog, 0)

    def compute_foil_loads(self, u_flow, rho, u_gamma=[]):

        u_flow = u_flow.reshape(-1,3)
        if u_gamma == []:
            u_main_wing = u_flow
            u_stab = u_flow
        else:
            u_main_wing = u_flow + u_gamma[0:self.main_wing.nsegs,:]
            u_stab = u_flow + u_gamma[self.main_wing.nsegs:(self.main_wing.nsegs+self.stabiliser.nsegs),:]

        main_wing_load = self.main_wing.LL_strip_theory_forces(u_main_wing, rho)
        stab_wing_load = self.stabiliser.LL_strip_theory_forces(u_stab, rho)
        mast_load = self.mast.LL_strip_theory_forces(u_flow, rho)

        total_load = np.sum(main_wing_load, axis=0) + np.sum(stab_wing_load, axis=0) + np.sum(mast_load, axis=0)

        main_wing_moment = np.cross((self.main_wing.xcp-self.cog.reshape(-1,3)), main_wing_load[:,0:3])
        stab_wing_moment = np.cross((self.stabiliser.xcp-self.cog.reshape(-1,3)), stab_wing_load[:,0:3])
        mast_moment = np.cross((self.mast.xcp - self.cog.reshape(-1, 3)), mast_load[:, 0:3])

        total_load[3:] = total_load[3:] + np.sum(main_wing_moment, axis=0) + np.sum(stab_wing_moment, axis=0) + np.sum(mast_moment, axis=0)
        return total_load

    def analyse_foil(self, angle, u_flow, rho, reflected_wake=False, wake_rollup=False, compare_strip=False, compare_roll_up=False):

        fig, (axL, axD, axM) = plt.subplots(3, 1)

        n_angles = angle.shape[0]
        n_speeds = u_flow.shape[0]
        if compare_strip:
            load_save_strip = np.zeros((angle.shape[0], 6, n_speeds))
        load_save_LL = np.zeros((angle.shape[0], 6, n_speeds))
        if compare_roll_up:
            load_save_LL_rollup = np.zeros((angle.shape[0], 6, n_speeds))
            wake_rollup=False

        for i in range(n_angles): # loop through angles
            # rotate foil to desired angle
            self.rotate_foil_assembly([angle[i], 0, 0])
            for j in range(n_speeds): # loop through speeds

                # compute LL forces on foil
                lifting_surfaces = self.surface2dict()
                out_LL = steady_LL_solve(lifting_surfaces, u_flow[j,:], rho, dt=0.1, min_dt=1, nit=50, reflected_wake=reflected_wake, wake_rollup=wake_rollup)
                u_cp = out_LL[0]
                loads_LL = self.compute_foil_loads(u_flow[j,:], rho, u_cp)
                load_save_LL[i,:,j] = loads_LL

                if compare_strip:
                    # compute strip theory forces on foil
                    loads_strip = self.compute_foil_loads(u_flow[j,:], rho)
                    load_save_strip[i,:, j] = loads_strip

                if compare_roll_up:
                    out_LL1 = steady_LL_solve(lifting_surfaces, u_flow[j,:], rho, dt=0.1, min_dt=1, nit=50, wake_rollup=True)
                    u_cp = out_LL1[0]
                    loads_LL = self.compute_foil_loads(u_flow[j,:], rho, u_cp)
                    load_save_LL_rollup[i,:,j] = loads_LL

            # rotate foil back to zero angle
            self.rotate_foil_assembly([-angle[i], 0, 0])
        
        for j in range(n_speeds):
            if np.max(load_save_LL[:,3,j]) > 0 and np.min(load_save_LL[:,3,j]) < 0:
                f = interp1d(load_save_LL[:,3,j], angle)
                angle_zero_moment = f(0)
                f = interp1d(angle, load_save_LL[:,2,j])
                lift_zero_moment = f(angle_zero_moment)
                f = interp1d(angle, load_save_LL[:,1,j])
                drag_zero_moment = f(angle_zero_moment)
                axM.plot(angle_zero_moment, 0, '*')
                axM.set_title('Zero moment pitch angle = ' + f'{angle_zero_moment:.3f}')
                axL.plot(angle_zero_moment, lift_zero_moment, '*')
                axL.set_title('Lift @ zero moment = ' + f'{lift_zero_moment:.2f}')
                axD.plot(angle_zero_moment, drag_zero_moment, '*')
                axD.set_title('Drag @ zero moment = ' + f'{drag_zero_moment:.2f}')

            if compare_roll_up:
                axL.plot(angle, load_save_LL_rollup[:,2,j], label='LL_rollup')
                axD.plot(angle, load_save_LL_rollup[:,1,j])
                axM.plot(angle, load_save_LL_rollup[:,3,j])
                
            if compare_strip:
                axL.plot(angle, load_save_strip[:,2,j], label='Strip')
                axD.plot(angle, load_save_strip[:,1,j])
                axM.plot(angle, load_save_strip[:,3,j])

            axL.plot(angle, load_save_LL[:,2,j], label='LL-V=' + f'{ms2knts(u_flow[j,1]):.1f}')
            axD.plot(angle, load_save_LL[:,1,j])
            axM.plot(angle, load_save_LL[:,3,j])

        axL.grid(True)
        axL.legend()
        axL.set_ylabel('Lift force (N)')
        axD.grid(True)
        axD.set_ylabel('Drag force (N)')
        axM.grid(True)
        axM.set_ylabel('Pitching moment (Nm)')
        axM.set_xlabel('Angle (deg)')


        plt.show()

    def surface2dict(self):
        fw = self.main_wing.create_dict()
        stab = self.stabiliser.create_dict()
        dict = [fw, stab]
        return dict

    def plot_wake(self, lifting_surfaces, wake_elmt_table, elmtIDs):
        # plot current foil assembly
        fig, ax = self.plot_foil_assembly()

        plot_wake(lifting_surfaces, wake_elmt_table, elmtIDs, ax=ax)

        span = self.main_wing.span   
        LE_forward_pt = np.max(self.main_wing.LE[:,1])
        LE_low_z_pt = np.min(self.main_wing.LE[:,2])
        ax.set_xlim3d(-span*1.1/2, span*1.1/2)
        ax.set_ylim3d(LE_forward_pt - span*2, LE_forward_pt)
        ax.set_zlim3d(LE_low_z_pt*1.1, LE_low_z_pt + 0.3)

class LiftingSurface:

    def __init__(self, rt_chord, span, Re=[], spline_pts=[], tip_chord=[], 
                 sweep_tip=0, sweep_curve=0, dih_tip=0, dih_curve=0,
                 washout_tip=0, washout_curve=0, afoil=[], afoil_path=[], 
                 Ncrit=9, type='wing', nsegs=50, units='mm', plot_flag=True):

        self.rt_chord = unit_2_meters(rt_chord, units)
        self.tip_chord = unit_2_meters(tip_chord, units)
        self.span = unit_2_meters(span, units)
        self.sweep_tip = unit_2_meters(sweep_tip, units)
        self.sweep_curve = sweep_curve
        self.dih_tip = unit_2_meters(dih_tip, units)
        self.dih_curve = dih_curve
        self.washout_tip = washout_tip
        self.washout_curve = washout_curve
        self.type = type
        self.afoil = afoil
        self.nsegs = nsegs
        self.a3 = None
        self.a2 = None
        self.a1 = None
        self.xcp = None
        self.qu_chord_loc = None
        self.ref_axis = None
        self.TE = None
        self.LE = None
        self.polar_re = None
        self.polar = None
        self.aerofoil_naca = []

        if spline_pts != []:
            self.generate_coords_spline(spline_pts, npts=1000, plot_flag=plot_flag)
        elif spline_pts == [] and tip_chord != []:
            self.generate_coords_simple(npts=1001)
        else:
            raise Exception("Either spline pts or tip chord must be prescribed.")

        self.polars_exist = False
        if afoil != []:
            self.define_aerofoil_geom(plot_flag=plot_flag)
            # self.compute_afoil_polar(angles=np.linspace(-5, 15, 21), Re=Re, plot_flag=plot_flag)
            try:
                self.compute_afoil_polar(angles=np.linspace(-5, 15, 21), Re=Re, Ncrit=Ncrit, plot_flag=plot_flag)
                self.polars_exist = True
            except:
                print("Aerofoil polar calculation failed.")
                self.polars_exist = False

        self.generate_LL_geom(nsegs, genBVs=True)

    def generate_coords_simple(self, npts=1001):
        if self.type == 'wing':
            self.x = np.linspace(-self.span / 2, self.span / 2, npts).reshape(npts, 1)
            self.f_chord = interp1d(np.array([-self.span / 2, 0, self.span / 2]),
                                    np.array([self.tip_chord, self.rt_chord, self.tip_chord]))
        elif self.type == 'mast':
            self.x = np.linspace(0, self.span, npts).reshape(npts, 1)
            self.f_chord = interp1d(np.array([0, self.span]),
                                    np.array([self.rt_chord, self.tip_chord]))

        sweep_curv = self.sweep_tip * (2 * np.abs(self.x) / self.span) ** self.sweep_curve
        dihedral_curve = self.dih_tip * (2 * np.abs(self.x) / self.span) ** self.dih_curve
        self.washout_curve = self.washout_tip * (2 * np.abs(self.x[:,0]) / self.span) ** self.washout_curve

        self.ref_axis = np.hstack((self.x, sweep_curv, dihedral_curve))

        chord = self.f_chord(self.x)

        dist_LE_2_refAxis = 0.5

        self.LE = np.hstack((self.x,
                             self.ref_axis[:, 1].reshape(npts, 1) + dist_LE_2_refAxis * chord,
                             dihedral_curve))
        self.TE = np.hstack((self.x,
                             self.ref_axis[:, 1].reshape(npts, 1) - (1 - dist_LE_2_refAxis) * chord,
                             dihedral_curve))
        if np.any(self.washout_curve != 0):
            c = np.cos(self.washout_curve*np.pi/180)
            s = np.sin(self.washout_curve*np.pi/180)
            self.LE[:,1] = (self.LE[:,1] - self.ref_axis[:,1]) * c - (self.LE[:,2] - self.ref_axis[:,2]) * s + self.ref_axis[:,1]
            self.LE[:,2] = (self.LE[:,1] - self.ref_axis[:,1]) * s + (self.LE[:,2] - self.ref_axis[:,2]) * c + self.ref_axis[:,2]
            self.TE[:,1] = (self.TE[:,1] - self.ref_axis[:,1]) * c - (self.TE[:,2] - self.ref_axis[:,2]) * s + self.ref_axis[:,1]
            self.TE[:,2] = (self.TE[:,1] - self.ref_axis[:,1]) * s + (self.TE[:,2] - self.ref_axis[:,2]) * c + self.ref_axis[:,2]

        self.qu_chord_loc = 0.75 * self.LE + 0.25 * self.TE

    def generate_coords_spline(self, spline_pts, plot_spline=False, plot_flag=False, npts=1000):
        x = spline_pts[:,0]
        x = np.concatenate((x[:-1], 
                            np.flip(x[1:]),
                            - x[:-1],
                            -np.flip(x[:])))
        y = np.concatenate((spline_pts[:-1,2],
                            np.flip(spline_pts[1:,1]),
                            spline_pts[:-1,1],
                            np.flip(spline_pts[:,2])))
        pts = np.stack((x * self.span/2, y * self.rt_chord)).T


        tck, u = splprep(pts.T, u=None, s=0.0, per=1, k=3) 
        u_new = np.linspace(u.min(), u.max(), 10000)
        x_new, y_new = splev(u_new, tck, der=0)
        

        if plot_spline:
            fig, ax = plt.subplots()
            ax.plot(pts[:,0], pts[:,1], 'ro')
            ax.plot(x_new, y_new, 'b-')
            ax.axis('scaled')
            ax.grid(True)
            plt.show()

        y_new = y_new - (np.max(y_new) + np.min(y_new)) / 2

        mask_LHS = x_new < 0
        mask_LE = y_new >= y_new[x_new == np.min(x_new)]
        mask_LE_RHS = y_new >= y_new[x_new == np.max(x_new)]
        mask_TE = y_new <= y_new[x_new == np.min(x_new)]
        mask_TE_RHS = y_new <= y_new[x_new == np.max(x_new)]

        LHS_LE = np.unique(np.stack((x_new[mask_LHS & mask_LE], y_new[mask_LHS & mask_LE])).T, axis=0)
        RHS_LE = np.unique(np.stack((x_new[~mask_LHS & mask_LE_RHS], y_new[~mask_LHS & mask_LE_RHS])).T, axis=0)
        LHS_TE = np.unique(np.stack((x_new[mask_LHS & mask_TE], y_new[mask_LHS & mask_TE])).T, axis=0)
        RHS_TE = np.unique(np.stack((x_new[~mask_LHS & mask_TE], y_new[~mask_LHS & mask_TE_RHS])).T, axis=0)
        LE = np.vstack((LHS_LE, RHS_LE))
        TE = np.vstack((LHS_TE, RHS_TE))
        self.x = np.linspace(np.min(x_new), np.max(x_new), 1000)
        LE = np.interp(self.x, LE[:,0], LE[:,1])
        TE = np.interp(self.x, TE[:,0], TE[:,1])

        dihedral_curv = self.dih_tip * (2 * np.abs(self.x) / self.span) ** self.dih_curve
        self.washout_curve = self.washout_tip * (2 * np.abs(self.x) / self.span) ** self.washout_curve
        self.LE = np.stack((self.x,
                             LE,
                             dihedral_curv)).T
        self.TE = np.stack((self.x,
                             TE,
                             dihedral_curv)).T

        self.ref_axis = (self.LE + self.TE) / 2
        if np.any(self.washout_curve != 0):
            c = np.cos(self.washout_curve*np.pi/180)
            s = np.sin(self.washout_curve*np.pi/180)
            self.LE[:,1] = (self.LE[:,1] - self.ref_axis[:,1]) * c - (self.LE[:,2] - self.ref_axis[:,2]) * s + self.ref_axis[:,1]
            self.LE[:,2] = (self.LE[:,1] - self.ref_axis[:,1]) * s + (self.LE[:,2] - self.ref_axis[:,2]) * c + self.ref_axis[:,2]
            self.TE[:,1] = (self.TE[:,1] - self.ref_axis[:,1]) * c - (self.TE[:,2] - self.ref_axis[:,2]) * s + self.ref_axis[:,1]
            self.TE[:,2] = (self.TE[:,1] - self.ref_axis[:,1]) * s + (self.TE[:,2] - self.ref_axis[:,2]) * c + self.ref_axis[:,2]
        
        self.qu_chord_loc = 0.75 * self.LE + 0.25 * self.TE

        if plot_flag:
            fig, ax = plt.subplots()
            ax.plot(self.x, self.washout_curve)
            # ax.axis('scaled')
            ax.grid(True)
            ax.set_ylim(-5,5)
            ax.set_xlabel("Span")
            ax.set_ylabel("Washout (deg)")
            plt.show()

        if plot_spline:
            self.calc_AR()

    def plot2D(self):
        x_coords = np.hstack((self.LE[:, 0], np.flip(self.LE[:, 0]), self.LE[0, 0]))
        y_coords = np.hstack((self.LE[:, 1], np.flip(self.TE[:, 1]), self.LE[0, 1]))

        fig, ax = plt.subplots()
        ax.plot(x_coords, y_coords, 'k-')  # plot external planform

        ax.plot(self.ref_axis[:, 0], self.ref_axis[:, 1], 'm-')  # plot ref axis
        ax.plot(self.qu_chord_loc[:, 0], self.qu_chord_loc[:, 1], 'g--')  # plot quarter chord

        ax.axis('scaled')
        ax.grid(True)
        plt.show()

    def plot3D(self, new_fig=True, fig=None, ax=None):
        x_coords = np.hstack((self.LE[:, 0], np.flip(self.TE[:, 0]), self.LE[0, 0]))
        y_coords = np.hstack((self.LE[:, 1], np.flip(self.TE[:, 1]), self.LE[0, 1]))
        z_coords = np.hstack((self.LE[:, 2], np.flip(self.TE[:, 2]), self.LE[0, 2]))

        if new_fig:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
        ax.plot3D(x_coords, y_coords, z_coords, 'k-')
        ax.plot3D(self.ref_axis[:, 0], self.ref_axis[:, 1], self.ref_axis[:, 2], 'm-')  # plot ref axis
        ax.plot3D(self.qu_chord_loc[:, 0], self.qu_chord_loc[:, 1], self.qu_chord_loc[:, 2],
                  'g--')  # plot quarter chord
        #         ax.axis('equal')
        if new_fig:
            lim = 1.1 * self.span / 2
            ax.set_xlim3d(-lim, lim)
            ax.set_ylim3d(-lim, lim)
            ax.set_zlim3d(-lim, lim)
            ax.grid(True)
            plt.show()
        else:
            return fig, ax

    def rotate_component(self, R):
        self.LE = apply_rotation(R, self.LE, 1)
        self.TE = apply_rotation(R, self.TE, 1)
        self.ref_axis = apply_rotation(R, self.ref_axis, 1)
        self.qu_chord_loc = apply_rotation(R, self.qu_chord_loc, 1)
        self.xcp = apply_rotation(R, self.xcp, 1)
        self.TEv = apply_rotation(R, self.TEv, 1)
        self.LEv = apply_rotation(R, self.LEv, 1)
        for i in range(len(self.BVs)):
            self.BVs[i].node1 = apply_rotation(R, self.BVs[i].node1, -1)
            self.BVs[i].node2 = apply_rotation(R, self.BVs[i].node2, -1)
            
        R_rotate_only = deepcopy(R)
        R_rotate_only[0:3, -1] = 0
        self.a1 = apply_rotation(R_rotate_only, self.a1, 1)
        self.a2 = apply_rotation(R_rotate_only, self.a2, 1)
        self.a3 = apply_rotation(R_rotate_only, self.a3, 1)
        self.dl = apply_rotation(R_rotate_only, self.dl, 1)

    def calc_simple_proj_wing_area(self):
        area = 0.5 * (self.rt_chord + self.tip_chord) * self.span
        print("Wing area is ", str(area*10000), " cm^2")
        return area

    def calc_proj_wing_area(self):
        area = np.trapz(np.linalg.norm(self.LE[:,:2] - self.TE[:,:2], axis=1), self.LE[:, 0])
        print("Projected wing area is ", str(area*10000), " cm^2")
        return area

    def calc_actual_wing_area(self):
        curved_ref_axis = np.append(0, np.cumsum(np.linalg.norm(np.diff(self.ref_axis, axis=0), axis=1)))
        area = np.trapz(np.linalg.norm(self.LE - self.TE, axis=1), curved_ref_axis)
        print("Actual wing area is ", str(area*10000), " cm^2")
        return area

    def calc_wing_volume(self):
        chord = np.linalg.norm(self.LE - self.TE, axis=1)
        curved_ref_axis = np.append(0, np.cumsum(np.linalg.norm(np.diff(self.ref_axis, axis=0), axis=1)))
        afoil_height = self.afoil_height_interpolator(self.t2c_distribution)
        areas = np.zeros(chord.shape)
        for i in range(len(chord)):
            areas[i] = np.trapz(chord[i] * afoil_height[:,1,i], chord[i] * afoil_height[:,0,i])
        volume = np.trapz(areas, curved_ref_axis)
        print("Wing volume is ", str(volume*1000000), " cm^3")
        return volume

    def calc_AR(self):
        AR = self.span ** 2 / self.calc_proj_wing_area()
        print("Aspect ratio is ", str(AR))
        return AR

    def define_aerofoil_geom(self, plot_flag=True):

        if plot_flag:
            fig, ax = plt.subplots()

        if type(self.afoil) is list:
            n_afoils = len(self.afoil)
            unique_afoil_names = pd.unique([afoil[0] for afoil in self.afoil]) # should be non-sorted
        else:
            n_afoils = 1
            self.afoil = [[self.afoil, 0]]

        # loop through unique afoils and generate/extract geometry
        self.afoil_table = {}
        for afoil in unique_afoil_names:
            # if airfoil defined as a naca aerofoil
            if "naca" in afoil:
                NACA=True
                # self.aerofoil_naca = self.afoil
                naca_numbers = afoil.replace("naca", "")
                max_camber = int(naca_numbers[0])
                camber_dist = int(naca_numbers[1])
                thick = int(naca_numbers[2:])

                # basic x y coords
                x_spacing = np.linspace(1, 0, 1000).reshape(-1, 1)
                x = x_spacing
                y = 5 * thick / 100 * (0.2969 * x ** 0.5 - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
                xU = x
                xL = np.flip(x)
                yU = y
                yL = - np.flip(y)

                # add camber
                yc = np.zeros(x.shape)
                if max_camber != 0:
                    m = max_camber / 100
                    p = camber_dist / 10
                    yc = np.zeros(x.shape)
                    mask = x <= p
                    yc[mask] = m / (p ** 2) * (2 * p * x[mask] - x[mask] ** 2)
                    yc[~mask] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[~mask] - x[~mask] ** 2)

                    dydx = np.zeros(x.shape)
                    dydx[mask] = 2 * m / p ** 2 * (p - x[mask])
                    dydx[~mask] = 2 * m / (1 - p) ** 2 * (p - x[~mask])
                    theta = np.arctan(dydx)
                    xU = xU - y * np.sin(theta)
                    xL = xL + np.flip(y * np.sin(theta))
                    yU = yc + y * np.cos(theta)
                    yL = np.flip(yc - y * np.cos(theta))

                x = np.vstack((xU, xL))
                y = np.vstack((yU, yL))
                xy = np.hstack((x,y))
                if plot_flag:
                    ax.plot(x_spacing, yc, '-.')

            else: # else if airfoil geometry given in a file
                
                # opening the CSV file
                with open(afoil, mode ='r') as file:
                    # reading the CSV file
                    csvFile = csv.reader(file, delimiter='\t')
                    # append contents of csv file
                    xy = []
                    for lines in csvFile:
                        try:
                            xy.append(list(map(float, lines)))
                        except:
                            line = lines[0].split()
                            xy.append(list(map(float, line)))

                # stack list of numbers
                xy = np.stack((xy), axis=0)
                x = xy[:,0]
                y = xy[:,1]

            # interpolate coords over new chord-wise grid
            LE_ID = np.argmin(xy[:,0])
            SS_mask = xy[:,1] >= xy[LE_ID,1]
            PS_mask = xy[:,1] <= xy[LE_ID,1]

            SS_coords = np.unique(xy[SS_mask,:], axis=0)
            PS_coords = np.unique(xy[PS_mask,:], axis=0)

            x_fine = np.linspace(xy[LE_ID,0], xy[np.argmax(xy[:,0]), 0], 1000)
            SS_coords_fine = np.interp(x_fine, SS_coords[:,0], SS_coords[:,1])
            PS_coords_fine = np.interp(x_fine, PS_coords[:,0], PS_coords[:,1])

            height = SS_coords_fine - PS_coords_fine

            # assign relevant airfoil outputs to a nested dictionary
            self.afoil_table[afoil] = {'coords': xy,
                                        'rel_thick': np.max(height),
                                        'norm_height': np.stack((x_fine, height), axis=1),
                                        'area': np.trapz(height, x_fine)}

            if plot_flag:
                ax.plot(x, y, '-', label=afoil)

        if plot_flag:
            ax.axis('scaled')
            ax.grid(True)
            ax.legend()
            plt.show()

        # if n_afoils > 1:
            # generate set of interpolated airfoil geometries
            # first generate control points for generating (interpolated) smooth t2c distribution
        afoil_spanwise_loc = np.stack([[-afoil[1], self.afoil_table[afoil[0]]['rel_thick']] for afoil in self.afoil])
        afoil_spanwise_loc = np.unique(np.vstack((np.flipud(afoil_spanwise_loc), np.abs(afoil_spanwise_loc))), axis=0)
        afoil_spanwise_loc[:,0] *= self.span/2

        # perform t2c interpolation
        self.t2c_interpolator = PchipInterpolator(afoil_spanwise_loc[:,0], afoil_spanwise_loc[:,1])
        self.t2c_distribution = self.t2c_interpolator(self.x)
        if plot_flag:
            fig, ax1 = plt.subplots()
            ax1.plot(self.x, self.t2c_distribution)
            ax1.grid(True)
            ax1.set_ylim(0, 0.4)
            ax1.set_xlabel('Span')
            ax1.set_ylabel('Relative thickness (-)')
            plt.show()

        if len(unique_afoil_names) > 1:
            # define airfoil cordinates interpolator
            rel_thick = [self.afoil_table[afoil]['rel_thick'] for afoil in self.afoil_table]
            self.afoil_coords_interpolator = interp1d(rel_thick,
                                    np.stack([self.afoil_table[afoil]['coords'] for afoil in self.afoil_table], axis=2))
            # define airfoil height interpolator  
            self.afoil_height_interpolator = interp1d(rel_thick,
                                                    np.stack([self.afoil_table[afoil]['norm_height'] for afoil in self.afoil_table], axis=2))
            # afoils = self.afoil_coords_interpolator(self.t2c_distribution)
        else:
            self.afoil_interpolator = lambda x: np.repeat(xy.reshape(-1,2,1), len(x), axis=2)
            self.afoil_height_interpolator = lambda x: np.repeat(np.stack((x_fine, height), axis=1).reshape(-1,2,1), len(x), axis=2)

    def compute_afoil_polar(self, angles, Re, Ncrit, plot_flag=False):
        direc = "afoil_polars/"
        if not os.path.isdir(direc):
            os.mkdir(direc)

        if plot_flag:
            fig, ax = plt.subplots()

        polar_table = np.nan * np.ones((angles.shape[0], 4, len(self.afoil_table)))
        # loop through unique airfoils in afoil_table
        for i, afoil in enumerate(self.afoil_table):
            # if afoil is naca, then xfoil doesn't need coordinates
            if "naca" in afoil:
                coeffs = find_coefficients(airfoil=afoil, alpha=angles, Reynolds=Re, Ncrit=Ncrit, iteration=1000, NACA=True, direc=direc)
            else: 
                # use coordinates in given file - not really sure this is working right now...
                coeffs = find_coefficients(airfoil=afoil, alpha=angles, Reynolds=Re, Ncrit=Ncrit, iteration=1000, NACA=False, GDES=False, direc=direc)
            
            # assemble afoil polar from xfoil coefficient data
            afoil_polar = np.hstack((np.array(coeffs["alpha"]).reshape(-1, 1),
                                      np.array(coeffs["CL"]).reshape(-1, 1),
                                      np.array(coeffs["CD"]).reshape(-1, 1),
                                      np.array(coeffs["CM"]).reshape(-1, 1)))
            # sometimes xfoil can miss points, so do linear interpolation
            interpolator = interp1d(afoil_polar[:,0], afoil_polar, axis=0, fill_value="extrapolate")
            afoil_polar = interpolator(angles)
            self.afoil_table[afoil]["Polar"] = afoil_polar

            # assemble polar table (3D)
            mask = np.isin(angles, afoil_polar[:,0])
            polar_table[mask,:,i] = afoil_polar

            if plot_flag:
                ax.plot(afoil_polar[:, 0], afoil_polar[:, 1], '-*', label='Cl ' + afoil)
                ax.plot(afoil_polar[:, 0], afoil_polar[:, 2], '-*', label='Cd ' + afoil)
                ax.plot(afoil_polar[:, 0], afoil_polar[:, 3], '-*', label='Cm ' + afoil)
                # ax.legend(['Cl', 'Cd', 'Cm'])
        
        if len(self.afoil_table) > 1:
            # define airfoil polar interpolator
            rel_thick = [self.afoil_table[afoil]['rel_thick'] for afoil in self.afoil_table]
            self.afoil_polar_interpolator = interp1d(rel_thick, polar_table)
        else:
            self.afoil_polar_interpolator = lambda x: np.repeat(polar_table.reshape(-1,4,1), len(x), axis=2)

        self.polar_re = Re
        if plot_flag:
            ax.grid(True)
            ax.set_xlabel('Angle of attack (deg)')
            ax.set_ylabel('Coeff (-)')
            ax.legend()
            plt.show()

    def calc_lift(self, V, aoa, rho):
        LE_elmt_ctr = (self.LE[0:-1, :] + self.LE[1:, :]) / 2
        TE_elmt_ctr = (self.TE[0:-1, :] + self.TE[1:, :]) / 2
        chord_elmt_ctr = np.linalg.norm(LE_elmt_ctr - TE_elmt_ctr, axis=1)
        dX = (self.LE[1:, 0] - self.LE[0:-1, 0])

        cl = np.stack([spline.__call__(aoa[i] * 180 / np.pi) for i, spline in enumerate(self.cl_splines)])
        # Simple flat plate lift coefficient 2*pi*aoa
        # cl = 2.0 * math.pi * aoa * math.pi / 180.0

        # compute lift per unit length @ elmt ctr
        # dL = 0.5*rho*V^2*c*cl
        dL_elmt_ctr = 0.5 * rho * V ** 2 * chord_elmt_ctr * cl

        # interpolate distributed lift at elmt nodes
        dL_nodes_f = interp1d(LE_elmt_ctr[:, 0], dL_elmt_ctr, fill_value="extrapolate")
        dL_nodes = dL_nodes_f(self.LE[:, 0])

        # Integrate lift w.r.t length
        # Note this is overly simplistic as it assumes lift from all strips acts perpendicular to the flow
        # This is not true for wings with dihedral/twist
        L = np.trapz(dL_nodes, self.LE[:, 0])
        return L

    def generate_LL_geom(self, n_segs, genBVs=False):
        # Employ cosine spacing for LL
        theta_ends = np.linspace(-math.pi / 2, math.pi / 2, n_segs + 1)
        seg_spacing = np.sin(theta_ends)
        theta_CP = (theta_ends[0:-1] + theta_ends[1:]) / 2
        seg_spacingCP = np.sin(theta_CP)
        #         seg_spacingCP = (seg_spacing[0:-1] + seg_spacing[1:])/2

        # Interpolate LE and TE on new segment spacing
        x_in = np.linspace(-1, 1, self.ref_axis.shape[0])
        LE_f = interp1d(x_in, self.LE, axis=0)
        LE_nodes = LE_f(seg_spacing)
        LE_CPs = LE_f(seg_spacingCP)
        self.LEv = LE_nodes
        TE_f = interp1d(x_in, self.TE, axis=0)
        TE_nodes = TE_f(seg_spacing)
        TE_CPs = TE_f(seg_spacingCP)
        self.TEv = TE_nodes

        # Define segment geometry points
        x1 = LE_nodes[0:-1, :]              # LE left
        x2 = TE_nodes[0:-1, :]              # TE left
        x3 = TE_nodes[1:, :]                # TE right
        x4 = LE_nodes[1:, :]                # LE right
        x5 = (x1 + x2) / 2                  # mid-chord left
        x7 = (x3 + x4) / 2                  # mid-chord right
        x6 = TE_CPs                         # TE mid-seg (cp)
        x8 = LE_CPs                         # LE mid-seg (cp)
        x9 = 0.75 * x1 + 0.25 * x2          # qu-chord left
        x10 = 0.75 * x4 + 0.25 * x3         # qu-chord right
        self.xcp = 0.75 * x8 + 0.25 * x6    # qu-chord mid-seg

        # Define segment normal vectors
        x6mx8 = x6 - x8
        x6mx8_norm = np.linalg.norm(x6mx8, axis=1).reshape(-1, 1)
        x10mx9 = x10 - x9
        x6mx8_cross_x10mx9 = np.cross(x6mx8, x10mx9)
        self.a1 = x6mx8 / x6mx8_norm
        self.a3 = x6mx8_cross_x10mx9 / np.linalg.norm(x6mx8_cross_x10mx9, axis=1).reshape(-1, 1)
        self.a2 = np.cross(self.a3, self.a1)
        self.dl = x10mx9

        # Compute segment area, chord, and sum total area
        self.dA = np.linalg.norm(np.cross(x6mx8, x7 - x5), axis=1).reshape(-1,1)
        self.c = x6mx8_norm
        self.LL_seg_area = np.sum(self.dA)

        # Define polars for each segment
        if self.polars_exist:
            t2c_CPs = self.t2c_interpolator(seg_spacingCP*self.span/2)
            afoil_polars_CPs = self.afoil_polar_interpolator(t2c_CPs)

            self.cl_splines = []
            self.cd_splines = []
            self.cm_splines = []
            for i, _ in enumerate(t2c_CPs):
                self.cl_splines.append(CubicSpline(afoil_polars_CPs[:,0,i], afoil_polars_CPs[:,1,i], axis=0))
                self.cd_splines.append(CubicSpline(afoil_polars_CPs[:,0,i], afoil_polars_CPs[:,2,i], axis=0))
                self.cm_splines.append(CubicSpline(afoil_polars_CPs[:,0,i], afoil_polars_CPs[:,3,i], axis=0))
            self.cl_tab = np.hstack((afoil_polars_CPs[:,0,0].reshape(-1,1), afoil_polars_CPs[:,1,:]))
            # self.cl_splines = jc.scipy.interpolate.InterpolatedUnivariateSpline(self.afoil_polar[:, 0], self.afoil_polar[:, 1])


        if genBVs:
            # Generate bound vorticity for this wing
            # BV order:
            # - All four elements from 1st segment, then 2nd, then 3rd, etc...
            # - LL, RHS, TE, LHS (i.e. clockwise round the segment, when looking from above)
            nodes1 = np.zeros((n_segs*4, 3)) # np.vstack((x9, x10, x3, x2))
            nodes2 = np.zeros((n_segs*4, 3)) # np.vstack((x10, x3, x2, x9))
            for i in range(n_segs):
                nodes1[4*i:4*(i+1),:] = np.vstack((x9[i,:], x10[i,:], x3[i,:], x2[i,:]))
                nodes2[4*i:4*(i+1),:] = np.vstack((x10[i,:], x3[i,:], x2[i,:], x9[i,:]))

            BVs = map(VortexLine, nodes1, nodes2)
            self.BVs = list(BVs)

    def LL_strip_theory_forces(self, u_motion, rho, full_output=True):
        u_cp = u_motion

        dot_ucp_a1 = np.sum(u_cp * self.a1, axis=1, keepdims=True)
        dot_ucp_a3 = np.sum(u_cp * self.a3, axis=1, keepdims=True)
        alpha_cp = np.arctan2(dot_ucp_a3, dot_ucp_a1)

        cl = np.stack([spline.__call__(alpha_cp[i] * 180 / np.pi) for i, spline in enumerate(self.cl_splines)])

        if full_output == False:
            lift_scalar = cl * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * self.dA
            return lift_scalar
        else:
            cd = np.stack([spline.__call__(alpha_cp[i] * 180 / np.pi) for i, spline in enumerate(self.cd_splines)])
            cm = np.stack([spline.__call__(alpha_cp[i] * 180 / np.pi) for i, spline in enumerate(self.cm_splines)])
            lift_scalar = cl * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * self.dA
            drag_scalar = cd * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * self.dA
            moment_scalar = cm * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * self.dA * self.c

            u_cp_norm = u_cp / np.linalg.norm(u_cp, axis=1, keepdims=True)
            cross_ucp_dl = np.cross(u_cp, self.dl)
            lift_norm = cross_ucp_dl/np.linalg.norm(cross_ucp_dl,axis=1,keepdims=True)
            lift_xyz = lift_scalar * lift_norm
            drag_xyz = drag_scalar * u_cp_norm
            moment_xyz = moment_scalar * self.a2

            force_xyz = lift_xyz + drag_xyz
            return np.hstack((force_xyz, moment_xyz))

    def create_dict(self):
        lifting_surface_dict = {"xcp": self.xcp,
                                "dl": self.dl,
                                "a1": self.a1,
                                "a3": self.a3,
                                "dA": self.dA,
                                "dl": self.dl,
                                "polar_alpha": self.cl_tab[:,0],
                                "polar_cl": self.cl_tab[:,1:],
                                "xnode1": np.concatenate([obj.node1.reshape(1, 1, -1) for obj in self.BVs], axis=1), # (1, nseg*4, 3)
                                "xnode2": np.concatenate([obj.node2.reshape(1, 1, -1) for obj in self.BVs], axis=1),
                                "TE": self.TEv,
                                "l0": np.array([obj.length0 for obj in self.BVs])}
        return lifting_surface_dict

    def export_wing_2_stl(self, stl_save_name, SF=1000, mounting_angle=0, resolution='low', plot_flag=True):

        # determine point spacing based on resolution request

        # compute arc length around root chord
        afoil_coords_cntr = self.afoil_table[self.afoil[0][0]]["coords"]
        _, indx = np.unique(afoil_coords_cntr, return_index=True, axis=0)
        afoil_coords_cntr = afoil_coords_cntr[np.sort(indx),:] * self.rt_chord
        s_coord_root = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(afoil_coords_cntr, axis=0) ** 2, axis=1))))
        arc_length_root = s_coord_root[-1]
        
        if resolution == 'high':
            spanwise_spacing = 0.0022   # 201 pts on stab440
            cs_spacing = 0.00073        # 251 cs pts on stab440
        elif resolution == 'medium':
            spanwise_spacing = 0.0044   # 101 pts on stab440
            cs_spacing = 0.0012         # 151 cs pts on stab440
        elif resolution == 'low':
            spanwise_spacing = 0.021    # 21 pts on stab440
            cs_spacing = 0.0073         # 25 cs pts on stab440

        ncs_pts = int(arc_length_root / cs_spacing)
        if (ncs_pts % 2) == 0:
            ncs_pts += 1 # if even, add 1
        
        ncs = int(self.span / spanwise_spacing)
        if (ncs % 2) == 0:
            ncs += 1 # if even, add 1
        print('Number of cross-sectional points = ', str(ncs_pts))
        print('Number of spanwise points = ', str(ncs))

        if plot_flag:
            fig, ax_afoil = plt.subplots()
        afoil_coords = np.zeros((ncs_pts, 3, len(self.afoil_table)))
        # loop through and pre-process afoils 
        for i, afoil in enumerate(self.afoil_table):
            # prep and interpolate airfoil on new grid
            # smoothing (pchip) interpolation of afoil on evenly spaced arc-length grid
            afoil_coords_in = self.afoil_table[afoil]["coords"]
            _, indx = np.unique(afoil_coords_in, return_index=True, axis=0)
            afoil_coords_in = afoil_coords_in[np.sort(indx),:]
            s_coord = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(afoil_coords_in, axis=0) ** 2, axis=1))))
            new_s_grid = np.linspace(s_coord[0], s_coord[-1], ncs_pts)
            # new_s_grid = cosspace(s_coord[0], s_coord[-1], n=-ncs_pts, factor=0.5)
            afoil_coords[:,1:,i] = pchip_interpolate(s_coord, afoil_coords_in, new_s_grid)

            if plot_flag:
                ax_afoil.plot(afoil_coords_in[:,0], afoil_coords_in[:,1], 'r-', label='Input')
                ax_afoil.plot(afoil_coords[:,1,i], afoil_coords[:,2,i], 'b-', marker=None, label='Interpolated')

            # Points that are close to TE are set to TE
            TE_mask = np.all(np.isclose(afoil_coords[:,:,i], [0,1,0]), axis=1)
            afoil_coords[TE_mask,:,i] = [0,1,0]
            # move airfoil centre to (0,0)
            afoil_coords[:,1,i] = afoil_coords[:,1,i] - 0.5
            # flip y axis so LE is +ve 0.5 in y
            afoil_coords[:,1,i] = - afoil_coords[:,1,i]

        if plot_flag:
            ax_afoil.legend()
            ax_afoil.axis('scaled')
            ax_afoil.grid(True)
            ax_afoil.set_title("2D Aerofoil interpolation")
            plt.show()

        # create afoil interpolator
        if len(self.afoil_table) > 1:
            rel_thick = [self.afoil_table[afoil]['rel_thick'] for afoil in self.afoil_table]
            afoil_coords_interpolator = interp1d(rel_thick, afoil_coords)
        else:
            afoil_coords_interpolator = lambda x: np.squeeze(afoil_coords)

        # interpolate LE, TE, ref_axis, t2c, washout on new spanwise grid
        # x_interp = np.linspace(self.x[0], self.x[-1], ncs)
        x_interp = cosspace(self.x[0], self.x[-1], n_pts=ncs)
        LEf = interp1d(self.x, self.LE, axis=0)
        LE = LEf(x_interp)
        TEf = interp1d(self.x, self.TE, axis=0)
        TE = TEf(x_interp)
        ref_axisf = interp1d(self.x, self.ref_axis, axis=0)
        ref_axis = ref_axisf(x_interp)
        t2cf = interp1d(self.x, self.t2c_distribution, axis=0)
        t2c = t2cf(x_interp)
        washoutf = interp1d(self.x, self.washout_curve, axis=0)
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
            coords1 = afoil_coords_interpolator(t2c[i])
            coords2 = afoil_coords_interpolator(t2c[i+1])
            coords1 = coords1 * chord[i] # afoil at station 1
            coords2 = coords2 * chord[i+1] # afoil at station 2

            # rotate normalised coordinates by washout (rotates about local (0,0))
            R1 = rotation_matrix([1,0,0], washout[i])
            R2 = rotation_matrix([1,0,0], washout[i+1])
            coords1 = apply_rotation(R1, coords1, dim=1)
            coords2 = apply_rotation(R2, coords2, dim=1)

            # rotate by anhedral?

            # shift normalised coordinates onto reference axis
            coords1 = ref_axis[i,:] + coords1
            coords2 = ref_axis[i+1,:] + coords2

            # rotate by mounting angle (rotates about global (0,0))
            if mounting_angle != 0:
                R1 = rotation_matrix([1,0,0], mounting_angle)
                R2 = rotation_matrix([1,0,0], mounting_angle)
                coords1 = apply_rotation(R1, coords1, dim=1)
                coords2 = apply_rotation(R2, coords2, dim=1)

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
        
        vertices = vertices*SF

        if plot_flag:
            # plot triangle surface
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            ax.plot_trisurf(vertices[:,0], vertices[:,1], faces, vertices[:,2])
            ax.set_xlim3d(-self.span*SF*1.1/2, self.span*SF*1.1/2)
            ax.set_ylim3d(-self.span*SF*1.1/2, self.span*SF*1.1/2)
            ax.set_zlim3d(-self.span*SF*1.1/2, self.span*SF*1.1/2)

        # generate stl mesh
        wing = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                wing.vectors[i][j] = vertices[int(f[j]),:]

        # Write the mesh to an stl file
        if '.stl' not in stl_save_name:
            stl_save_name = stl_save_name + '.stl'
        print('Saving stl with name:', stl_save_name)
        wing.save(stl_save_name)

class EllipticalWing(LiftingSurface):
    # This child class is used for verification of the lifting line model
    def __init__(self, rt_chord, span, Re, sweep_tip=0, sweep_curv=0, dih_tip=0, dih_curve=0,
                 afoil = [], nsegs=50, units='mm'):

        self.rt_chord = unit_2_meters(rt_chord, units)
        self.span = unit_2_meters(span, units)
        self.sweep_tip = unit_2_meters(sweep_tip, units)
        self.sweep_curv = sweep_curv
        self.dih_tip = unit_2_meters(dih_tip, units)
        self.dih_curve = dih_curve
        self.nsegs = nsegs
        self.a3 = None
        self.a2 = None
        self.a1 = None
        self.xcp = None
        self.polar_re = None
        self.polar = None
        self.afoil = afoil
        
        npts = 10001
        self.x = np.linspace(-self.span / 2, self.span / 2, npts).reshape(npts, 1)
        self.ref_axis = np.hstack((self.x, np.zeros((npts, 2))))

        self.chord_ini = rt_chord*np.sqrt(1 - (self.x / (span / 2)) ** 2)

        self.LE = np.hstack((self.x,
                             self.ref_axis[:, 1].reshape(npts, 1) + 0.25 * self.chord_ini,
                             self.ref_axis[:, 2].reshape(npts, 1)))
        self.TE = np.hstack((self.x,
                             self.ref_axis[:, 1].reshape(npts, 1) - 0.75 * self.chord_ini,
                             self.ref_axis[:, 2].reshape(npts, 1)))

        self.qu_chord_loc = 0.75 * self.LE + 0.25 * self.TE

        self.polars_exist = False
        if afoil != []:
            self.define_aerofoil(afoil, False)
            self.compute_afoil_polar(angles=np.linspace(-5, 15, 21), Re=Re, plot_flag=False)
            self.polars_exist = True

        self.generate_LL_geom(nsegs, genBVs=True)

    def define_flat_plate_polar(self):
        self.cl_tab = np.stack((np.linspace(-20,20,100), np.linspace(-20,20,100)*np.pi/180*2*np.pi), axis=1)
        self.cl_tab = np.hstack((self.cl_tab[:,0].reshape(-1,1), np.repeat(self.cl_tab[:,1].reshape(-1,1), self.nsegs, axis=1)))
        self.cl_splines = [CubicSpline(self.cl_tab[:, 0], self.cl_tab[:, 1])] * self.nsegs
        self.polars_exist = True

class VortexLine:
    vortex_elmtID = 0

    def __init__(self, node1, node2, vtype='bound'):
        self.node1 = node1
        self.node2 = node2
        self.length0 = np.linalg.norm(node2 - node1)

        self.vtype = vtype
        self.circ = 0

        self.elmtID = VortexLine.vortex_elmtID
        VortexLine.vortex_elmtID += 1
