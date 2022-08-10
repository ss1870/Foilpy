#%%
from foilpy.splines import BSplineCurve, parameterise_curve, distribute_knots, curve_interp
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
        super().__init__(p, U)

        self.q = q
        self.V  = V
        self.knotsV = np.array(self.V[self.q:-self.q])
        self.s = len(V) - 1
        self.m = self.s - self.q - 1
        self.M_v = self.def_m_matrix(self.knotsV, self.q)
        if np.any(P != None):
            self.contrl_pts = P # Control points
            assert P.shape[0] - 1 == self.r - self.p - 1
            assert P.shape[1] - 1 == self.s - self.q - 1
            self.ndims = self.contrl_pts.shape[2]
            self.npts_u = P.shape[0]
            self.npts_v = P.shape[1]

    def eval_surf(self, u, v):
        """
        Evaluate spline surface at (u, v) coordinate.
        """
        # Find knot spans for u and v coords
        span_u = self.find_span2(self.knotsU, u)
        span_v = self.find_span2(self.knotsV, v)

        s = u - self.knotsU[span_u]
        s_vec_u = np.array([s**3, s**2, s, 1])
        s = v - self.knotsV[span_v]
        s_vec_v = np.array([s**3, s**2, s, 1])

        cpID_u = [span_u, span_u+4]
        if cpID_u[0] < 0:
            cpID_u = [0, 4]
        if cpID_u[1] > self.npts_u - 1:
            cpID_u = [self.npts_u-4, self.npts_u]
        cpID_v = [span_v, span_v+4]
        if cpID_v[0] < 0:
            cpID_v = [0, 4]
        if cpID_v[1] > self.npts_v - 1:
            cpID_v = [self.npts_v-4, self.npts_v]

        span = self.find_span(u)
        N = self.basis_funs(span, u)

        Nu = np.matmul(s_vec_u, self.M_u[:,:,span_u]).reshape(4,1)
        Nv = np.matmul(s_vec_v, self.M_v[:,:,span_v]).reshape(4,1)
        P_temp = self.contrl_pts[cpID_u[0]:cpID_u[1],cpID_v[0]:cpID_v[1],:]
        S = np.zeros((self.ndims))
        for i in range(self.ndims):
            S[i] = np.matmul(np.matmul(Nu.T, P_temp[:,:,i]), Nv)
        return S

    def grid_plot(self, scatter_pts=None, npts=100):
        fig, ax = plt.subplots()
        ax = fig.add_subplot(projection='3d')
        u = np.linspace(0,1,npts)
        v = np.linspace(0,1,npts)
        isogrid_pts = np.zeros((npts,npts,self.ndims))
        for i in range(len(u)):
            for j in range(len(v)):
                isogrid_pts[i,j,:] = self.eval_surf(u[i],v[j])
            ax.plot3D(isogrid_pts[i,:,0], isogrid_pts[i,:,1], isogrid_pts[i,:,2], color='black')
        for j in range(len(v)):
            ax.plot3D(isogrid_pts[:,j,0], isogrid_pts[:,j,1], isogrid_pts[:,j,2], color='black')
        if np.any(scatter_pts != None):
            ax.scatter(scatter_pts[:,:,0],scatter_pts[:,:,1],scatter_pts[:,:,2],marker='o')


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


def surf_interp(Q, p, q, u_bar=None, v_bar=None, U=None, V=None, 
                param_method='centripetal', plot_flag=True):
    """
    Interpolate surface.
    """
    npts_u = Q.shape[0]
    npts_v = Q.shape[1]
    nkts_u = npts_u + p + 1
    nkts_v = npts_v + q + 1

    # Parameterise surface of points
    if np.any(u_bar == None) or np.any(v_bar == None):
        u_bar, v_bar = parameterise_surf(points, method=param_method)

    # Compute knot vector
    if np.any(U == None) or np.any(V == None):
        U = distribute_knots(u_bar, p, nkts_u, method='even_interp')
        V = distribute_knots(v_bar, q, nkts_v, method='even_interp')

    ## Interpolate surface pts
    # Loop through v direction, interpolate each curve for fixed v
    P_interp1 = np.zeros((npts_u, npts_v, 3))
    for i in range(len(v_bar)):
        temp_crv = curve_interp(points[:,i,:], p, u_bar, U, plot_flag=False)
        P_interp1[:,i,:] = temp_crv.contrl_pts

    # Loop through u direction, interpolate each set of CPs for fixed u
    P_interp2 = np.zeros((npts_u, npts_v, 3))
    for j in range(len(u_bar)):
        temp_crv = curve_interp(P_interp1[j,:,:], q, v_bar, V, plot_flag=False)
        P_interp2[j,:,:] = temp_crv.contrl_pts

    surf = SplineSurface(p,q,U,V,P_interp2)
    
    if plot_flag:
        surf.grid_plot(scatter_pts=Q, npts=20)
    return surf


points = (((-5, -5, 0), (-2.5, -5, 0), (0, -5, 0), (2.5, -5, 0), (5, -5, 0), (7.5, -5, 0), (10, -5, 0)),
          ((-5, 0, 3), (-2.5, 0, 3), (0, 0, 3), (2.5, 0, 3), (5, 0, 3), (7.5, 0, 3), (10, 0, 3)),
          ((-5, 5, 0), (-2.5, 5, 0), (0, 5, 0), (2.5, 5, 0), (5, 5, 0), (7.5, 5, 0), (10, 5, 0)),
          ((-5, 7.5, -3), (-2.5, 7.5, -3), (0, 7.5, -3), (2.5, 7.5, -3), (5, 7.5, -3), (7.5, 7.5, -3), (10, 7.5, -3)),
          ((-5, 10, 0), (-2.5, 10, 0), (0, 10, 0), (2.5, 10, 0), (5, 10, 0), (7.5, 10, 0), (10, 10, 0)))


%matplotlib widget
points = np.asarray(points)
p = 3
q = 3

surf = surf_interp(points, p, q, param_method='Fang', plot_flag=True)





## Test
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
# Data set
points = ((-5, -5, 0), (-2.5, -5, 0), (0, -5, 0), (2.5, -5, 0), (5, -5, 0), (7.5, -5, 0), (10, -5, 0),
          (-5, 0, 3), (-2.5, 0, 3), (0, 0, 3), (2.5, 0, 3), (5, 0, 3), (7.5, 0, 3), (10, 0, 3),
          (-5, 5, 0), (-2.5, 5, 0), (0, 5, 0), (2.5, 5, 0), (5, 5, 0), (7.5, 5, 0), (10, 5, 0),
          (-5, 7.5, -3), (-2.5, 7.5, -3), (0, 7.5, -3), (2.5, 7.5, -3), (5, 7.5, -3), (7.5, 7.5, -3), (10, 7.5, -3),
          (-5, 10, 0), (-2.5, 10, 0), (0, 10, 0), (2.5, 10, 0), (5, 10, 0), (7.5, 10, 0), (10, 10, 0))
size_u = 5
size_v = 7
degree_u = 3
degree_v = 3

surf = fitting.interpolate_surface(points, size_u, size_v, degree_u, degree_v)

# Plot the interpolated surface
surf.delta = 0.05
surf.vis = vis.VisSurface()
surf.render()

pts = np.array(surf.ctrlpts2d)
mysurf = SplineSurface(degree_u,degree_v,surf.knotvector_u,surf.knotvector_v,pts)
assert np.all(np.isclose(mysurf.eval_surf(0.2,0.5), surf.evaluate_single([0.2,0.5])))
mysurf.grid_plot(npts=20)