from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
import math
from source.AeroPy.aeropy.xfoil_module import find_coefficients
from source.LL_functions import rotation_matrix, translation_matrix, apply_rotation
import jax_cosmo as jc


def ms2knts(velocity):
    return velocity * 1.943844

def knts2ms(velocity):
    return velocity / 1.943844

def unit_2_meters(val, unit):
    if unit=='mm':
        return val/1000
    elif unit == 'cm':
        return val/100
    elif unit == 'm':
        return val

class FoilAssembly:

    def __init__(self, main_wing0, stabiliser0, mast0, fuselage_length, mast_attachment_ratio, wing_angle=0,
                 stabiliser_angle=0, units='mm'):
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
        R_mast = np.dot(translation_matrix([0, 0, -self.mast.span]), rotation_matrix([0, 1, 0], -90))
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
        rider_mass = 72 # 72
        rider_cog = np.array([0, 0.385, 0.900]) # np.array([0, 385, 900])

        self.total_mass = self.mast0.mass + self.main_wing0.mass + self.stabiliser0.mass + fuselage_mass + board_mass + rider_mass
        cog = (self.mast0.mass*self.mast0.cog + self.main_wing0.mass*self.mast0.cog + self.stabiliser0.mass*self.mast0.cog + fuselage_mass*fuselage_cog + board_mass*board_cog + rider_mass*rider_cog)/self.total_mass
        self.cog = cog.reshape(3,1)
        print("Total mass = ", str(self.total_mass), "\n")
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

        total_load[3:] = total_load[3:] + np.sum(main_wing_moment, axis=0)  + np.sum(stab_wing_moment, axis=0)  + np.sum(mast_moment, axis=0) 
        return total_load

    def surface2dict(self):
        fw = self.main_wing.create_dict()
        
        #  {"xcp": self.main_wing.xcp, 
        #       "dl": self.main_wing.dl, 
        #       "a1": self.main_wing.a1, 
        #       "a3": self.main_wing.a3, 
        #       "dA": self.main_wing.dA, 
        #       "dl": self.main_wing.dl,
        #     #   "cl_spl": self.main_wing.cl_spline,
        #       "cl_tab": self.main_wing.cl_tab,
        #       "xnode1": np.concatenate([obj.node1.reshape(1, 1, -1) for obj in self.main_wing.BVs], axis=1), # (1, nseg*4, 3)
        #       "xnode2": np.concatenate([obj.node2.reshape(1, 1, -1) for obj in self.main_wing.BVs], axis=1),
        #       "TE": self.main_wing.TEv, 
        #       "l0": np.array([obj.length0 for obj in self.main_wing.BVs])} 
        
        stab = self.stabiliser.create_dict()
        # {"xcp": self.stabiliser.xcp, 
        #         "dl": self.stabiliser.dl, 
        #         "a1": self.stabiliser.a1, 
        #         "a3": self.stabiliser.a3, 
        #         "dA": self.stabiliser.dA, 
        #         "dl": self.stabiliser.dl,
        #         # "cl_spl": self.stabiliser.cl_spline,
        #         "cl_tab": self.stabiliser.cl_tab,
        #         "xnode1": np.concatenate([obj.node1.reshape(1, 1, -1) for obj in self.stabiliser.BVs], axis=1), # (1, nseg*4, 3)
        #         "xnode2": np.concatenate([obj.node2.reshape(1, 1, -1) for obj in self.stabiliser.BVs], axis=1),
        #         "TE": self.stabiliser.TEv, 
        #         "l0": np.array([obj.length0 for obj in self.stabiliser.BVs])}
        dict = [fw, stab]
        return dict


class LiftingSurface:

    def __init__(self, rt_chord, tip_chord, span, Re, sweep_tip=0, sweep_curv=0, dih_tip=0, dih_curve=0,
                 afoil_name='naca0012', type='wing', nsegs=50, units='mm'):

        self.rt_chord = unit_2_meters(rt_chord, units)
        self.tip_chord = unit_2_meters(tip_chord, units)
        self.span = unit_2_meters(span, units)
        self.sweep_tip = unit_2_meters(sweep_tip, units)
        self.sweep_curv = sweep_curv
        self.dih_tip = unit_2_meters(dih_tip, units)
        self.dih_curve = dih_curve
        self.type = type
        self.afoil_name = afoil_name
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
        

        self.generate_coords(npts=101)
        if afoil_name != []:
            self.define_aerofoil(afoil_name, False)
            self.compute_afoil_polar(angles=np.linspace(-5, 15, 21), Re=Re, plot_flag=False)
        # # front_wing.plot2D()
        # # front_wing.plot3D()
        # print("Front wing area =", str(front_wing.calc_simple_proj_wing_area()))
        # print("Trapz front wing area =", str(front_wing.calc_trapz_proj_wing_area()))
        # print("Front wing aspect ratio =", str(front_wing.calc_AR()))
        # print("Front wing lift =", str(front_wing.calc_lift(V=5, aoa=0, rho=1025)), "Newtons")
        self.generate_LL_geom(nsegs, genBVs=True)
        # print("Lifting line front wing area =", str(front_wing.LL_seg_area))

    def generate_coords(self, npts=1001):
        if self.type == 'wing':
            self.x = np.linspace(-self.span / 2, self.span / 2, npts).reshape(npts, 1)
            self.f_chord = interp1d(np.array([-self.span / 2, 0, self.span / 2]),
                                    np.array([self.tip_chord, self.rt_chord, self.tip_chord]))
        elif self.type == 'mast':
            self.x = np.linspace(0, self.span, npts).reshape(npts, 1)
            self.f_chord = interp1d(np.array([0, self.span]),
                                    np.array([self.rt_chord, self.tip_chord]))

        sweep_curv = self.sweep_tip * (2 * np.abs(self.x) / self.span) ** self.sweep_curv
        dihedral_curv = self.dih_tip * (2 * np.abs(self.x) / self.span) ** self.dih_curve

        self.ref_axis = np.hstack((self.x, sweep_curv, dihedral_curv))

        chord = self.f_chord(self.x)

        dist_LE_2_refAxis = 0.5

        self.LE = np.hstack((self.x,
                             self.ref_axis[:, 1].reshape(npts, 1) + dist_LE_2_refAxis * chord,
                             dihedral_curv))
        self.TE = np.hstack((self.x,
                             self.ref_axis[:, 1].reshape(npts, 1) - (1 - dist_LE_2_refAxis) * chord,
                             dihedral_curv))

        self.qu_chord_loc = 0.75 * self.LE + 0.25 * self.TE

        # Attempt at smoothing wing tips - incomplete
        #     mid_tip = 0.5 * (self.LE[0,-1] + self.TE[0,-1])
        #     tip_smoothing_param = 0.8
        #     x_id = int(tip_smoothing_param*npts)
        #     mask = np.hstack((np.arange(0, x_id) , npts-1))
        #     x_in = self.x[0, mask]
        #     LE_in = np.hstack((self.LE[0, mask[0:-1]], self.LE[0, -1] - (1-tip_smoothing_param) * self.tip_chord))
        #     LE_new = pchip_interpolate(x_in, LE_in, self.x)
        #     print(LE_new)

    def plot2D(self):
        x_coords = np.hstack((self.LE[:, 0], np.flip(self.LE[:, 0]), self.LE[0, 0]))
        y_coords = np.hstack((self.LE[:, 1], np.flip(self.TE[:, 1]), self.LE[0, 1]))

        plt.plot(x_coords, y_coords, 'k-')  # plot external planform

        plt.plot(self.ref_axis[:, 0], self.ref_axis[:, 1], 'm-')  # plot ref axis
        plt.plot(self.qu_chord_loc[:, 0], self.qu_chord_loc[:, 1], 'g--')  # plot quarter chord

        plt.axis('scaled')
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
        return area

    def calc_trapz_proj_wing_area(self):
        area = np.trapz(np.linalg.norm(self.LE - self.TE, axis=1), self.LE[:, 0])
        return area

    def calc_AR(self):
        AR = self.span ** 2 / self.calc_trapz_proj_wing_area()
        return AR

    def define_aerofoil(self, naca, plot_flag=True):
        self.aerofoil_naca = naca
        naca_numbers = naca.replace("naca", "")
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
        self.afoil_coords = np.hstack((x, y))

        if plot_flag:
            plt.plot(x_spacing, yc, '-.')
            plt.plot(x, y, '-')
            plt.grid(True)
            plt.axis('scaled')
            plt.show()

    def compute_afoil_polar(self, angles, Re, plot_flag=False):
        coeffs = find_coefficients(airfoil=self.aerofoil_naca, alpha=angles, Reynolds=Re, iteration=1000)

        self.afoil_polar = np.hstack((np.array(coeffs["alpha"]).reshape(-1, 1),
                                      np.array(coeffs["CL"]).reshape(-1, 1),
                                      np.array(coeffs["CD"]).reshape(-1, 1),
                                      np.array(coeffs["CM"]).reshape(-1, 1)))
        self.polar_re = Re

        # self.cl_spline = UnivariateSpline(self.afoil_polar[:, 0], self.afoil_polar[:, 1], s=0.1)
        # self.cd_spline = UnivariateSpline(self.afoil_polar[:, 0], self.afoil_polar[:, 2], s=0.001)
        # self.cm_spline = UnivariateSpline(self.afoil_polar[:, 0], self.afoil_polar[:, 3], s=0.0001)
        self.cl_spline1 = CubicSpline(self.afoil_polar[:, 0], self.afoil_polar[:, 1])
        self.cd_spline = CubicSpline(self.afoil_polar[:, 0], self.afoil_polar[:, 2])
        self.cm_spline = CubicSpline(self.afoil_polar[:, 0], self.afoil_polar[:, 3])

        self.cl_spline = jc.scipy.interpolate.InterpolatedUnivariateSpline(self.afoil_polar[:, 0], self.afoil_polar[:, 1])
        self.cl_tab = np.hstack((self.afoil_polar[:,0].reshape(-1,1), self.afoil_polar[:,1].reshape(-1,1)))

        if plot_flag:
            plt.plot(self.afoil_polar[:, 0], self.afoil_polar[:, 1], '-')
            # plt.plot(np.linspace(angles[0],angles[-1],100), self.cl_spline(np.linspace(angles[0],angles[-1],100)))
            plt.plot(self.afoil_polar[:, 0], self.afoil_polar[:, 2], '-')
            plt.plot(self.afoil_polar[:, 0], self.afoil_polar[:, 3], '-')
            plt.grid(True)
            plt.xlabel('Angle of attack (deg)')
            plt.ylabel('Coeff (-)')
            plt.legend(['Cl', 'Cd', 'Cm'])
            plt.show()

    def calc_lift(self, V, aoa, rho):
        LE_elmt_ctr = (self.LE[0:-1, :] + self.LE[1:, :]) / 2
        TE_elmt_ctr = (self.TE[0:-1, :] + self.TE[1:, :]) / 2
        chord_elmt_ctr = np.linalg.norm(LE_elmt_ctr - TE_elmt_ctr, axis=1)
        dX = (self.LE[1:, 0] - self.LE[0:-1, 0])

        cl = self.cl_spline.__call__(aoa)
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

        cl = self.cl_spline1.__call__(alpha_cp * 180 / np.pi)

        if full_output == False:
            lift_scalar = cl * 0.5 * rho * (dot_ucp_a1 ** 2 + dot_ucp_a3 ** 2) * self.dA
            return lift_scalar
        else:
            cd = self.cd_spline.__call__(alpha_cp * 180 / np.pi)
            cm = self.cm_spline.__call__(alpha_cp * 180 / np.pi)
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
                                #   "cl_spl": self.main_wing.cl_spline,
                                # "cl_tab": self.cl_tab,
                                "polar_alpha": self.cl_tab[:,0],
                                "polar_cl": np.tile(self.cl_tab[:,1].reshape(-1,1), (1, len(self.dA))),
                                "xnode1": np.concatenate([obj.node1.reshape(1, 1, -1) for obj in self.BVs], axis=1), # (1, nseg*4, 3)
                                "xnode2": np.concatenate([obj.node2.reshape(1, 1, -1) for obj in self.BVs], axis=1),
                                "TE": self.TEv, 
                                "l0": np.array([obj.length0 for obj in self.BVs])} 
        return lifting_surface_dict

class EllipticalWing(LiftingSurface):
    # This child class is used for verification of the lifting line model
    def __init__(self, rt_chord, span, Re, sweep_tip=0, sweep_curv=0, dih_tip=0, dih_curve=0,
                 afoil_name='naca0012', nsegs=50, units='mm'):

        self.rt_chord = unit_2_meters(rt_chord, units)
        self.span = unit_2_meters(span, units)
        self.sweep_tip = unit_2_meters(sweep_tip, units)
        self.sweep_curv = sweep_curv
        self.dih_tip = unit_2_meters(dih_tip, units)
        self.dih_curve = dih_curve
        self.afoil_name = afoil_name
        self.nsegs = nsegs
        self.a3 = None
        self.a2 = None
        self.a1 = None
        self.xcp = None
        self.polar_re = None
        self.polar = None
        
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

        if afoil_name != []:
            self.define_aerofoil(afoil_name, False)
            self.compute_afoil_polar(angles=np.linspace(-5, 15, 21), Re=Re, plot_flag=False)

        self.generate_LL_geom(nsegs, genBVs=True)

    def define_flat_plate_polar(self):
        self.cl_tab = np.stack((np.linspace(-20,20,100), np.linspace(-20,20,100)*np.pi/180*2*np.pi), axis=1)
        self.cl_spline1 = CubicSpline(self.cl_tab[:, 0], self.cl_tab[:, 1])

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
