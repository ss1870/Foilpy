"""
This module contains guesses/approximations of some of the AXIS components.
These are by no means accurate or representative.
"""

import numpy as np
from foilpy.classes import LiftingSurface

def bsc_810(re, nsegs=40, plot_flag=False):
    """
    AXIS Broad Spectrum Carve (BSC) 810
    """
    span = 0.81
    max_chrd = 0.155
    # probably has 1-3 degrees of nose-down washout (means root stalls earlier than tip)
    coords = np.array([[0,   0,      1],
                    [0.5, 0.01,   0.9],
                    [0.7, 0.02,   0.77],
                    [0.9, 0.03,    0.53],
                    [0.98,0.08,     0.36],
                    [1,   0.12,    0.12]])

    afoil = [['naca2215', 0],
            ['naca2215', (0.055/2)/(span/2)],
            ['naca1712', (0.15/2)/(span/2)],
            ['naca1712', 1]]

    bsc_810 = LiftingSurface(rt_chord=max_chrd,
                                spline_pts=coords,
                                span=span,
                                Re=re,
                                dih_tip=-0.03,
                                dih_curve=2,
                                washout_tip=-3,
                                washout_curve=4,
                                afoil=afoil,
                                nsegs=nsegs,
                                units='m',
                                plot_flag=plot_flag)

    if plot_flag:
        bsc_810.plot2D()
        bsc_810.plot3D()
        bsc_810.calc_wing_volume()

    # assert following
    
    assert np.isclose(bsc_810.calc_AR(), 6.42, rtol=1e-03, atol=1e-03)
    assert np.isclose(bsc_810.calc_proj_wing_area()*10000, 1022, rtol=1e-03, atol=1e-03)
    # assert np.isclose(bsc_810.calc_actual_wing_area()*10000, 1070, rtol=1e-03, atol=1e-03)
    # assert np.isclose(bsc_810.calc_wing_volume()*1000000, 1284, rtol=1e-03, atol=1e-03)
    return bsc_810

def stab_fr_440(re, nsegs=40, plot_flag=False):
    """
    AXIS Freeride stabiliser 440
    """
    span = 0.440
    max_chrd = 0.090

    afoil = [['naca0012', 0],
             ['naca0012', 1]]
    # probably has 1-3 degrees of nose-down washout (means root stalls earlier than tip)
    coords = np.array([[0,  0,     1],
                    [0.35,  0.00,   0.92],
                    [0.8,   0.00,   0.57],
                    [0.98,  0.05,   0.24],
                    [1,     0.08,   0.08]])
    stab_fr_440 = LiftingSurface(rt_chord=max_chrd,
                                spline_pts=coords,
                                span=span,
                                Re=re,
                                dih_tip=0.02,
                                dih_curve=8,
                                afoil=afoil,
                                nsegs=nsegs,
                                units='m',
                                plot_flag=plot_flag)

    if plot_flag:
        stab_fr_440.plot2D()
        stab_fr_440.plot3D()

    # assert following
    # assert np.isclose(stab_fr_440.calc_AR(), 6.37, rtol=1e-03, atol=1e-03)
    # assert np.isclose(stab_fr_440.calc_proj_wing_area()*10000, 303.97, rtol=1e-03, atol=1e-03)
    # assert np.isclose(stab_fr_440.calc_actual_wing_area()*10000, 318.94, rtol=1e-03, atol=1e-03)
    # assert np.isclose(stab_fr_440.calc_wing_volume()*1000000, 192.81, rtol=1e-03, atol=1e-03)
    return stab_fr_440

def mast_75cm(re, nsegs=8, plot_flag=False):
    """
    AXIS 19mm 75cm mast
    """
    span = 0.75
    chord = 0.13

    afoil = [['naca0015', 0],
             ['naca0015', 1]] # 19mm + 130mm chord ~15% thickness
    mast_75cm = LiftingSurface(rt_chord=chord,
                          tip_chord=chord,
                          span=span,
                          Re=re,
                          type='mast',
                          afoil=afoil,
                          nsegs=nsegs,
                          units='m',
                          plot_flag=plot_flag)

    if plot_flag:
        mast_75cm.plot2D()
        mast_75cm.plot3D()

    # assert following
    # assert np.isclose(Stab_FR_440.calc_AR(), 6.37, rtol=1e-03, atol=1e-03)
    # assert np.isclose(Stab_FR_440.calc_proj_wing_area()*10000, 303.97, rtol=1e-03, atol=1e-03)
    # assert np.isclose(Stab_FR_440.calc_actual_wing_area()*10000, 318.94, rtol=1e-03, atol=1e-03)
    # assert np.isclose(Stab_FR_440.calc_wing_volume()*1000000, 192.81, rtol=1e-03, atol=1e-03)
    return mast_75cm
