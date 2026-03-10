#%%
import foilpy.splines.curve as fsp
import AXIS_wing_definitions as AX_wings
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from copy import deepcopy
%matplotlib widget

# Run a quick test to check that scaling the control points of a spline by
# chord results in the same shape as scaling the points
SCALE = 2.3 # scaling factor

RE = 5 * 0.2 * 1025 / 0.00126
wing = AX_wings.bsc_810(RE, nsegs=40, plot_flag=False)
## Define coords and non-dimensional arc length u
coords = wing.afoil_table['naca1214']['coords']
coords = np.delete(coords,999,axis=0)
s = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))))
norm_s = s / s[-1]
interper = interp1d(norm_s, coords, axis=0)

# Define 2D spline curve of an aerofoil
curve = fsp.curve_approx(coords, 31, 3, u_bar=None, U=None, plot_flag=True, 
                        knot_spacing='adaptive', param_method='Fang')
# Generate points from spline
pts = curve.eval_list(norm_s) * SCALE

# Create new spline, scale control points and generate pts
curve1 = deepcopy(curve)
curve1.Pw[:,:-1] *= SCALE
pts1 = curve1.eval_list(norm_s)

fig, ax = plt.subplots()
ax.plot(pts[:,0], pts[:,1], 'k-')
ax.plot(pts1[:,0], pts1[:,1], 'r--')
ax.plot(curve1.Pw[:,0], curve1.Pw[:,1], '.', markersize=10, color='red', label='Scaled Control Points')
# ax.plot(pts[:,0] - pts1[:,0], pts[:,1] - pts1[:,1], 'r--')
ax.axis('scaled')
ax.grid(True)
