#%%

import numpy as np

drag = 35

rider_weight = 75

alpha1 = 40 # side to side when looking directly downwind at window
alpha2 = 20 # radial angle downwind when looking at window from side
alpha_f = 15 # angle of foil mast when looking from side

# x = side to side 
# y = down wind/upwind
# z = vertical

# first assume drag (+ bit of wind resistance) == x component from kite
F_kite = drag*1.1 / np.cos(alpha2*np.pi/180) / np.sin(alpha1*np.pi/180)
F_kite_y = F_kite * np.sin(alpha2*np.pi/180) 
F_kite_z = F_kite * np.cos(alpha2*np.pi/180) * np.cos(alpha1*np.pi/180) 


F_foil = F_kite_y / np.sin(alpha_f*np.pi/180)


# vertical equilibrium
F_foil_z = rider_weight*9.81 - F_kite_z

# take foil z component back into line of the mast
F_foil1 = F_foil_z / np.cos(alpha_f*np.pi/180) 

F_foil1 * np.sin(alpha_f*np.pi/180) 


