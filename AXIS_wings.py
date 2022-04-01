#%%

%matplotlib widget
from source.classes import LiftingSurface, knts2ms
import numpy as np
from source.LL_functions import steady_LL_solve, plot_wake

# AXIS Broad Spectrum Carve (BSC) 810
span = 0.81
max_chrd = 0.155
re = 5 * max_chrd * 1026 / 0.00126 # u=5m/s, rho=1026 
afoil = 'hq109' # this has around 9/10% thickness
# probably has 1-3 degrees of nose-down washout (means root stalls earlier than tip)
coords = np.array([[0,   0,      1],
                   [0.5, 0.01,   0.91],
                   [0.7, 0.02,   0.79],
                   [0.9, 0.03,    0.56],
                   [0.98,0.08,     0.36],
                   [1,   0.12,    0.12]])
BSC_810 = LiftingSurface(rt_chord=max_chrd, 
                            spline_pts=coords, 
                            span=span,
                            Re=re,
                            dih_tip=-0.03,  
                            dih_curve=2,    
                            afoil=afoil, 
                            nsegs=40,
                            units='m')

BSC_810.plot2D()
BSC_810.plot3D()

BSC_810.calc_proj_wing_area()
BSC_810.calc_AR()


# assert following
assert np.isclose(BSC_810.calc_AR(), 6.42, rtol=1e-03, atol=1e-03)
assert np.isclose(BSC_810.calc_proj_wing_area()*10000, 1022, rtol=1e-03, atol=1e-03)
assert np.isclose(BSC_810.calc_actual_wing_area()*10000, 1070, rtol=1e-03, atol=1e-03)
# assert np.isclose(BSC_810.calc_wing_volume()*1000000, 1284, rtol=1e-03, atol=1e-03)

# To do: 
# - implement functionality for splined shapes
# - calc wing volume
# - add option for variable relative thickness from root to tip (10 -> 8%?)
# - add option for washout
