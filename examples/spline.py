#%%
import foilpy.splines.curve as fsp
import AXIS_wing_definitions as AX_wings
import numpy as np
from scipy.interpolate import interp1d
import importlib
importlib.reload(fsp)
%matplotlib widget

RE = 5 * 0.2 * 1025 / 0.00126
wing = AX_wings.bsc_810(RE, nsegs=40, plot_flag=False)
## Define coords and non-dimensional arc length u
coords = wing.afoil_table['naca1214']['coords']
coords = np.delete(coords,999,axis=0)
s = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))))
norm_s = s / s[-1]
interper = interp1d(norm_s, coords, axis=0)

## Loop through different numbers of control points/curve orders

## Inputs - no control points and degree
ncp = 21        # no of control points
p = 3           # degree of curve

# Inferred curve properties
n = ncp - 1
m = n + p + 1
nk = m + 1      # no of knots

# First do linear solve of interpolation to get a starting point
Q = interper(np.linspace(0,1,ncp))
curve = fsp.curve_interp(Q, p, plot_flag=True)

curve = fsp.curve_approx(coords, 31, p, u_bar=None, U=None, plot_flag=True, 
                        knot_spacing='adaptive', param_method='Fang')

a=rgbrhj
## set up optimisation

## Wrapper
# inputs: control points, weights, knot vector, degree
opti_options = dict(
    optCPs=True,
    optWeights=False,
    optKnots=True,
    U = curve.U,
    weights=curve.weights,
    nCPs=ncp,
    p=p,
    n=n,
    m=m,
    ndims=curve.ndims,
    pt_list=np.hstack((coords, norm_s.reshape(-1,1)))
)

X0 = []
LB = []
UB = []
DVinfo = []
if opti_options["optCPs"]:
    for i in range(curve.ndims):
        X0.extend((curve.contrl_pts[:,i] - -2) / (2 - -2))
        LB.extend([0.0]*ncp)
        UB.extend([1.0]*ncp)
        DVinfo.extend([[1,i]]*ncp)
if opti_options["optWeights"]:
    X0.extend([1]*ncp)
    LB.extend([0.001]*ncp)
    UB.extend([10]*ncp)
    DVinfo.extend([[2,0]]*ncp)
if opti_options["optKnots"]:
    X0.extend(curve.knots[1:-1])
    LB.extend([0.0]*(nk-8))
    UB.extend([1.0]*(nk-8))
    DVinfo.extend([[3,0]]*(nk-8))

opti_options["DVinfo"] = np.array(DVinfo)
X0 = np.array(X0)
LB = np.array(LB)
UB = np.array(UB)

nDVs = len(X0)
lin_con_coeffs = np.zeros((nDVs,nDVs))
lin_con_lb = -np.inf * np.ones((nDVs))
lin_con_ub = np.inf * np.ones((nDVs))
knt_counter = 0
for i in range(nDVs):
    if opti_options["DVinfo"][i,0] == 3:
        lin_con_coeffs[i,i] = 1
        if knt_counter < (nk-8-1):
            lin_con_coeffs[i,i+1] = -1
            lin_con_ub[i] = 0
        else:
            lin_con_ub[i] = 1
        knt_counter += 1


F0 = fsp.wrapper(X0, opti_options)

F = lambda X : fsp.wrapper(X, opti_options)

from scipy.optimize import Bounds, LinearConstraint, minimize
bounds = Bounds(LB, UB)
lin_constr = LinearConstraint(lin_con_coeffs, lin_con_lb, lin_con_ub)

from scipy.optimize import SR1
# jac="2-point", hess=SR1(),
res = minimize(F, X0, method='trust-constr', jac="3-point",
               constraints=[lin_constr],
               options={'verbose': 1, 'maxiter': 20, 'disp': True}, 
               bounds=bounds)

print(F(X0), F(res.x))

opti_curve = fsp.curve_from_DVs(res.x, opti_options)

random_ind = np.linspace(0, len(opti_options["pt_list"])-1, 101).astype(int)
opti_curve.plot_curve(method=2, extra_pts=coords[random_ind, :])
# res = minimize(F, res.x, method='trust-constr',
#                constraints=[lin_constr],
#                options={'verbose': 1, 'maxiter': 20, 'disp': True}, 
#                bounds=bounds)

# To try:
# - projection method - might be faster as well? With small random selection?/ADAM optimiser
# - run an approximation at each evaluation to get control points, optimise knot vector and number of CPs
    # - Need to implement approximation algorithm