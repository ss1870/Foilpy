
#%%
import AXIS_wing_definitions as AX_wings
# %matplotlib widget


# wing = AX_wings.BSC_810(813492, afoil='hq109', nsegs=40, plot_flag=True)
front_wing = AX_wings.BSC_810(813492, afoil='naca1710', nsegs=40, plot_flag=True)
# wing = AX_wings.BSC_810(1e6, afoil='naca2413', nsegs=40, plot_flag=True)
