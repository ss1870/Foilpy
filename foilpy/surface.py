#%%
from pytest import param
from foilpy.splines import BSplineCurve
import numpy as np
import matplotlib.pyplot as plt


class SplineSurface(BSplineCurve):
    """
    Child class of BSplineCurve - adds V dimension to existing U parameterisation.
    Nomenclature:
        - Surface parameterised on u and v
        - Control points P_i,j
            - i = 0,...,n, j = 0,...,m
            - npts in u is n+1, npts in v is m+1
        - u direction:
            - degree = p
            - knot vector = U = [[0]*(p+1), u_p+1, ..., u_r-p-1, [1]*(p+1)]
            - no knots = r+1, where r = n+p+1
        - v direction:
            - degree = q
            - knot vector = V = [[0]*(q+1), v_q+1, ..., v_s-q-1, [1]*(q+1)]
            - no knots = s+1, where s = m+q+1
    """
    def __init__(self, p, q, U, V, P=None):
        super().__init__(p, U, P=P)

        self.q = q
        self.V  = V
        self.knotsV = np.array(self.V[self.q:-self.q])
        self.s = len(V) - 1
        self.m = self.s - self.q - 1




def parameterise_surf(points, method='centripetal'):
    """
    Parameterises a surface of points with u and v coordinates.
    Takes average of curve parameterisations in each direction.
    points dimensions = (npts_u, npts_v, ndim)
    """
    ncp_u = points.shape[0]
    ncp_v = points.shape[1]
    # loop through each u-curve
    u_bar = np.zeros((ncp_u, ncp_v))
    for i in range(ncp_v):
        u_bar[:,i] = parameterise_curve(points[:,i,:], method=method)
    u_bar = np.mean(u_bar, axis=1)
    # loop through each v-curve
    v_bar = np.zeros((ncp_v, ncp_u))
    for i in range(ncp_u):
        v_bar[:,i] = parameterise_curve(points[i,:,:], method=method)
    v_bar = np.mean(v_bar, axis=1)
    return u_bar, v_bar

def define_knots(u_bar, nkts, p, method='equal'):
    """
    For a given set of parameterised coordinates, define a knot vector.
    Knot vector is used for interpolation/approximation.
    """
    U = np.zeros((nkts))
    U[:p+1] = 0
    U[-1-p:] = 1
    if method == 'equal':
        U[p+1:-1-p] = [np.sum(u_bar[j:j+p])/p for j in range(1,len(u_bar)-p)]
    else: 
        print("Method unrecognised.")
    return U

points = (((-5, -5, 0), (-2.5, -5, 0), (0, -5, 0), (2.5, -5, 0), (5, -5, 0), (7.5, -5, 0), (10, -5, 0)),
          ((-5, 0, 3), (-2.5, 0, 3), (0, 0, 3), (2.5, 0, 3), (5, 0, 3), (7.5, 0, 3), (10, 0, 3)),
          ((-5, 5, 0), (-2.5, 5, 0), (0, 5, 0), (2.5, 5, 0), (5, 5, 0), (7.5, 5, 0), (10, 5, 0)),
          ((-5, 7.5, -3), (-2.5, 7.5, -3), (0, 7.5, -3), (2.5, 7.5, -3), (5, 7.5, -3), (7.5, 7.5, -3), (10, 7.5, -3)),
          ((-5, 10, 0), (-2.5, 10, 0), (0, 10, 0), (2.5, 10, 0), (5, 10, 0), (7.5, 10, 0), (10, 10, 0)))

points = np.asarray(points)
npts_u = points.shape[0]
npts_v = points.shape[1]
p = 2
q = 3

# Parameterise surface of points
u_bar, v_bar = parameterise_surf(points, 'Fang')

# Determine knot vectors U, V - for interpolation
nkts_u = npts_u + p + 1
nkts_v = npts_v + q + 1
U = define_knots(u_bar, nkts_u, p, method='equal')
V = define_knots(v_bar, nkts_v, q, method='equal')
