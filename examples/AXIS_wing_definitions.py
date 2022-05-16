from foilpy.classes import LiftingSurface
import numpy as np

def BSC_810(re, afoil='hq109', nsegs=40, plot_flag=False):
    # AXIS Broad Spectrum Carve (BSC) 810
    span = 0.81
    max_chrd = 0.155
    # probably has 1-3 degrees of nose-down washout (means root stalls earlier than tip)
    coords = np.array([[0,   0,      1],
                    [0.5, 0.01,   0.9],
                    [0.7, 0.02,   0.77],
                    [0.9, 0.03,    0.53],
                    [0.98,0.08,     0.36],
                    [1,   0.12,    0.12]])
    BSC_810 = LiftingSurface(rt_chord=max_chrd, 
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
        BSC_810.plot2D()
        BSC_810.plot3D()

    # assert following
    assert np.isclose(BSC_810.calc_AR(), 6.42, rtol=1e-03, atol=1e-03)
    assert np.isclose(BSC_810.calc_proj_wing_area()*10000, 1022, rtol=1e-03, atol=1e-03)
    # assert np.isclose(BSC_810.calc_actual_wing_area()*10000, 1070, rtol=1e-03, atol=1e-03)
    # assert np.isclose(BSC_810.calc_wing_volume()*1000000, 1284, rtol=1e-03, atol=1e-03)
    return BSC_810

def Stab_FR_440(re, nsegs=40, plot_flag=False):
    # AXIS Freeride stabiliser 440
    span = 0.440
    max_chrd = 0.090
    # re = 5 * max_chrd * 1026 / 0.00126 # u=5m/s, rho=1026 
    afoil = 'naca0012' # this has around 9/10% thickness hq109
    # probably has 1-3 degrees of nose-down washout (means root stalls earlier than tip)
    coords = np.array([[0,  0,     1],
                    [0.35,  0.00,   0.92],
                    [0.8,   0.00,   0.57],
                    [0.98,  0.05,   0.24],
                    [1,     0.08,   0.08]])
    Stab_FR_440 = LiftingSurface(rt_chord=max_chrd, 
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
        Stab_FR_440.plot2D()
        Stab_FR_440.plot3D()

    # assert following
    # assert np.isclose(Stab_FR_440.calc_AR(), 6.37, rtol=1e-03, atol=1e-03)
    # assert np.isclose(Stab_FR_440.calc_proj_wing_area()*10000, 303.97, rtol=1e-03, atol=1e-03)
    # assert np.isclose(Stab_FR_440.calc_actual_wing_area()*10000, 318.94, rtol=1e-03, atol=1e-03)
    # assert np.isclose(Stab_FR_440.calc_wing_volume()*1000000, 192.81, rtol=1e-03, atol=1e-03)
    return Stab_FR_440

