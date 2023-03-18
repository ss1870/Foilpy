from foilpy.foildef import FoilAssembly
from foilpy.LL_functions import steady_LL_solve, plot_wake
from foilpy.utils import ms2knts
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def analyse_foil(foilAssembly: FoilAssembly,
                angle, u_flow, rho, reflected_wake=False, wake_rollup=False, 
                compare_strip=False, compare_roll_up=False):

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
        foilAssembly.rotate_foil_assembly([angle[i], 0, 0])
        for j in range(n_speeds): # loop through speeds

            # compute LL forces on foil
            lifting_surfaces = foilAssembly.surface2dict()
            out_LL = steady_LL_solve(lifting_surfaces, u_flow[j,:], rho, dt=0.1, min_dt=1, nit=50, reflected_wake=reflected_wake, wake_rollup=wake_rollup)
            u_cp = out_LL[0]
            loads_LL = foilAssembly.compute_foil_loads(u_flow[j,:], rho, u_cp)
            load_save_LL[i,:,j] = loads_LL

            if compare_strip:
                # compute strip theory forces on foil
                loads_strip = foilAssembly.compute_foil_loads(u_flow[j,:], rho)
                load_save_strip[i,:, j] = loads_strip

            if compare_roll_up:
                out_LL1 = steady_LL_solve(lifting_surfaces, u_flow[j,:], rho, dt=0.1, min_dt=1, nit=50, wake_rollup=True)
                u_cp = out_LL1[0]
                loads_LL = foilAssembly.compute_foil_loads(u_flow[j,:], rho, u_cp)
                load_save_LL_rollup[i,:,j] = loads_LL

        # rotate foil back to zero angle
        foilAssembly.rotate_foil_assembly([-angle[i], 0, 0])
    
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


def plot_wake(foilAssembly: FoilAssembly,
                lifting_surfaces, wake_elmt_table, elmtIDs):
    # plot current foil assembly
    fig, ax = foilAssembly.plot_foil_assembly()

    plot_wake(lifting_surfaces, wake_elmt_table, elmtIDs, ax=ax)

    span = foilAssembly.main_wing.span   
    LE_forward_pt = np.max(foilAssembly.main_wing.LE[:,1])
    LE_low_z_pt = np.min(foilAssembly.main_wing.LE[:,2])
    ax.set_xlim3d(-span*1.1/2, span*1.1/2)
    ax.set_ylim3d(LE_forward_pt - span*2, LE_forward_pt)
    ax.set_zlim3d(LE_low_z_pt*1.1, LE_low_z_pt + 0.3)

