#%%
%matplotlib widget

import AXIS_wing_definitions as AXIS



wing = AXIS.BSC_810(1e6, afoil='../afoil_geom/hq109', nsegs=40, plot_flag=True)

