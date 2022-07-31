#%%
from this import d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

class BSplineCurve():
    """
    B-Spline/NURBS curve class.
    """
    def __init__(self, p, U, w=None, P=None):
        self.degree = p     # Degree
        self.U = U          # Knot vector
        self.knots = np.array(self.U[self.degree:-self.degree])
        self.m = len(U) - 1 # No of knots = m + 1
        self.n = self.m - self.degree - 1
        self.Np = self.n + 1
        self.M = None
        self.def_M()
        if np.any(P != None):
            self.contrl_pts = P # Control points
            n = len(P) - 1 # No of control points = n+1
            assert n == self.m - self.degree - 1
            self.n = n
            self.ndims = self.contrl_pts.shape[1]
            if np.any(w != None):
                # if weights are present then this is a NURBS curve
                self.weights = w    # Weights
                self.Pw = np.hstack((self.weights * self.contrl_pts, self.weights))
        
    def def_M(self):
        """
        Define series of M matrices for each spline segment.
        Used for evaluating points and derivatives.
        http://and-what-happened.blogspot.com/2012/07/evaluating-b-splines-aka-basis-splines.html
        """
        if self.M is None and self.degree == 3:
            nseg = len(self.knots) - 1
            kv = np.append(np.append([self.knots[0]]*2, self.knots), [self.knots[-1]]*2)

            M = np.zeros((4,4,nseg))
            for i in range(nseg):
                v0 = kv[i] - kv[i+2]
                v1 = kv[i+1] - kv[i+2]
                v3 = kv[i+3] - kv[i+2]
                v4 = kv[i+4] - kv[i+2]
                v5 = kv[i+5] - kv[i+2]

                a = 1 / (v3 * (v3 - v1) * (v3 - v0))
                b = 1 / (v3 * (v3 - v1) * (v4 - v1))
                c = 1 / (v3 * v4 * (v4 - v1))
                d = 1 / (v3 * v4 * v5)

                M[0,0,i] = -a
                M[0,1,i] = a + b + c
                M[0,2,i] = -b-c-d
                M[0,3,i] = d
                M[1,0,i] = v3 * a * 3
                M[1,1,i] = -(((v3+v3+v0)*a)+((v3+v4+v1)*b)+((v4+v4)*c))
                M[1,2,i] = (((v1+v1+v3)*b)+((v1+v4)*c)+(v5*d))
                M[1,3,i] = 0
                M[2,0,i] = -(v3*v3*a*3)
                M[2,1,i] = (((v3+v0+v0)*v3*a)+(((v3*v4)+((v3+v4)*v1))*b)+(v4*v4*c))
                M[2,2,i] = -((((v1+v3+v3)*b)+(v4*c))*v1)
                M[2,3,i] = 0
                M[3,0,i] = (v3*v3*v3*a)
                M[3,1,i] = -(((v3*v0*a)+(v4*v1*b))*v3)
                M[3,2,i] = (v1*v1*v3*b)
                M[3,3,i] = 0

            self.M = M

    def find_span(self, u): # A2.1
        """
        Determine the knot span index
        """
        if u > self.U[-1]:
            raise Exception("Query u must be less than U[end](=%d)"%self.U[-1])
        if u < self.U[0]:
            raise Exception("Query u must be greater than U[0](=%d)"%self.U[0])
        if u == self.U[self.n+1]:
            return self.n
        low = self.degree
        high = self.n + 1
        mid = int((low + high) / 2)
        while u < self.U[mid] or u >= self.U[mid+1]:
            if u < self.U[mid]:
                high = mid
            else:
                low = mid
            mid = int((low + high) / 2)
        return mid

    def basis_funs(self, i, u, ders=0): # A2.2
        """
        Compute the non-vanishing basis functions
        """
        N = [1.0]
        left = np.zeros((self.degree+1))
        right = np.zeros((self.degree+1))
        calc_ders = False
        if ders > 0:
            n = ders
            calc_ders = True
        if calc_ders:
            ndu = np.zeros((self.degree+1, self.degree+1))
            ndu[0,0] = 1
        for j in range(1, self.degree+1):
            left[j] = u - self.U[i+1-j]
            right[j] =  self.U[i+j] - u
            saved = 0.0
            if calc_ders: 
                saved1 = 0.0
            for r in range(j):
                temp1 = (right[r+1] + left[j-r])
                temp = N[r] / temp1
                N[r] = saved + right[r+1] * temp
                saved = left[j-r]*temp
                if calc_ders:
                    ndu[j,r] = temp1
                    temp2 = ndu[r,j-1] / ndu[j,r]
                    ndu[r,j] = saved1 + right[r+1] * temp
                    saved1 = left[j-r] * temp2
            N.append(saved)
            if calc_ders:
                ndu[j,j] = saved1
        if calc_ders:
            ders = np.zeros((n+1, self.degree+1))
            for j in range(self.degree+1):
                ders[0,j] = ndu[j,self.degree]
            a = np.zeros((n+1, self.degree+1))
            for r in range(self.degree+1):
                s1=0
                s2=1
                a[0,0] = 1
                for k in range(1, n+1):
                    d = 0.0
                    rk = r - k
                    pk = self.degree - k
                    if r >= k:
                        a[s2,0] = a[s1,0] / ndu[pk+1,rk]
                        d = a[s2,0] * ndu[rk,pk]
                    if rk >= -1:
                        j1 = 1
                    else:
                        j1 = -rk
                    if r <= pk+1:
                        j2 = k-1
                    else:
                        j2 = self.degree - r
                    for j in range(j1, j2+1):
                        a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1,rk+j]
                        d += a[s2,j] * ndu[rk+j,pk]
                    if r <= pk:
                        a[s2,k] = -a[s1,k-1] / ndu[pk+1,r]
                        d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j = s1
                    s1 = s2
                    s2 = j
            r = self.degree
            for k in range(1,n+1):
                for j in range(self.degree+1):
                    ders[k,j] *= r
                    r *= self.degree - k

        if calc_ders:
            return np.array(N).reshape(-1,1), ders
        else:
            return np.array(N).reshape(-1,1)

    def one_basis_fun(self, p, i, u):
        """
        Compute Nip for a single basis function
        """
        if (i == 0 and u == self.U[0]) or (i == self.m-p-1 and u ==self.U[self.m]):
            Nip = 1.0
            return Nip
        if u < self.U[i] or u >= self.U[i+p+1]:
            Nip = 0.0
            return Nip
        N = np.zeros((p+1))
        for j in range(p+1):
            if u >= self.U[i+j] and u < self.U[i+j+1]:
                N[j] = 1.0
        for k in range(1,p+1):
            if N[0] == 0.0:
                saved = 0.0
            else:
                saved = ((u-self.U[i]) * N[0]) / (self.U[i+k] - self.U[i])
            for j in range(p-k+1):
                Uleft = self.U[i+j+1]
                Uright = self.U[i+j+k+1]
                if N[j+1] == 0.0:
                    N[j] = saved
                    saved = 0.0
                else:
                    temp = N[j+1] / (Uright - Uleft)
                    N[j] = saved + (Uright - u) * temp
                    saved = (u - Uleft) * temp
        Nip = N[0]
        return Nip

    def eval_curve(self, u, method=2, der1=False, der2=False):
        """
        Evaluate curve at single point u on non-dimensional arc
        """
        if (der1 or der2) and method==1:
            raise Exception("Error: derivative options only for eval method=2.")

        if method == 1:
            span = self.find_span(u)
            N = self.basis_funs(span, u)
            Cw = np.sum(np.tile(N, (1,self.ndims+1)) * self.Pw[span-self.degree:span+1,:], axis=0)
            
        elif method == 2 and self.degree==3:
            # find which span u is in
            temp = self.knots < u
            if np.all(temp == False):
                span = 0
            else:
                span = np.where(self.knots < u)[0][-1]

            s = u - self.knots[span]
            s_vec = np.array([s**3, s**2, s, 1])

            cpID = [span, span+4]
            if cpID[0] < 0:
                cpID = [0, 4]
            if cpID[1] > self.Np - 1:
                cpID = [self.Np-4, self.Np]
            MCw = np.matmul(self.M[:,:,span], self.Pw[cpID[0]:cpID[1],:])
            Cw = np.matmul(s_vec, MCw)

        C = Cw[:self.ndims]/Cw[self.ndims]
        if der1:
            # first derivative w.r.t u parameter
            s_vecd1 = np.array([3*s**2, 2*s, 1, 0])
            dCw1 = np.matmul(s_vecd1, MCw)
            dC1 = dCw1[:self.ndims]/dCw1[self.ndims]
        if der2:
            # second derivative w.r.t u parameter
            s_vecd2 = np.array([6*s, 2, 0, 0])
            dCw2 = np.matmul(s_vecd2, MCw)
            dC2 = dCw2[:self.ndims]/dCw2[self.ndims]

        return C

    def eval_der(self, u, d):
        """
        Alg3.2: Evaluate derivate of B spline at u.
        u = parameter location
        d = derivative (1 = 1st, 2 = 2nd)
        """
        if hasattr(self, 'weights'):
            raise Exception("Evaluating derivative not work for NURBS.")
        du = min(d, self.degree)
        CK = np.zeros((du+1, 1))
        for k in range(self.degree+1, d+1):
            CK[k,:] = 0
        span = self.find_span(u)
        N, nders = self.basis_funs(span, u, ders=d)
        for k in range(du+1):
            CK[k,:] = 0
            for j in range(self.degree+1):
                CK[k,:] += nders[k,j] #* self.contrl_pts[span-self.degree+j,:]

    def eval_list(self, u, method=2):
        """
        Evaluate curve at a list of u parameters.
        """
        C = np.zeros((len(u), self.ndims))
        for i, u in enumerate(u):
            C[i] = self.eval_curve(u, method=method)
        return C

    def def_mapping(self, npts=1000, plot_flag=False):
        # evaluate curve
        u = np.linspace(self.U[0], self.U[-1], npts)
        xy = self.eval_list(u, method=2)

        # compute arc-length coords
        s = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))))
        self.length = s[-1]

        # define interper mappings
        self.u2s = interp1d(u, s)
        self.u2norms = interp1d(u, s/self.length)
        self.s2u = interp1d(s, u)
        self.norms2u = interp1d(s/self.length, u)

        if plot_flag:
            fig, ax = plt.subplots()
            ax.plot(u/u[-1], s/self.length)
            ax.axis('scaled')
            ax.set_xlabel('Norm u')
            ax.set_ylabel('Norm s')

    def plot_basis_funs(self, plot_flag=True):
        """
        Plot all basis functions on u grid between U[0] and U[-1]
        """
        u_all = np.linspace(self.U[0], self.U[-1], 1000)
        N = np.zeros((self.n+1, len(u_all)))
        for j, u in enumerate(u_all):
            for i in range(self.n+1):
                N[i,j] = self.one_basis_fun(self.degree, i, u)

        plt.plot(u_all, N.T)
        return N
    
    def plot_curve(self, pts=100, method=2, return_axes=False, 
                    extra_pts=None):
        """
        Plot curve over full u domain
        """
        u_all = np.linspace(self.U[0], self.U[-1], 1000)
        C = self.eval_list(u_all, method=method)

        if self.ndims == 2:
            fig, ax = plt.subplots()
            ax.plot(C[:,0], C[:,1], label='Spline curve')
            ax.plot(self.contrl_pts[:,0], self.contrl_pts[:,1], linestyle='--', marker='.', label='CPs')
            if np.any(extra_pts != None):
                ax.plot(extra_pts[:,0], extra_pts[:,1], linestyle='', marker='.')
            ax.axis('scaled')
        elif self.ndims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot3D(C[:,0], C[:,1], C[:,2], label='Spline curve')
            ax.plot3D(self.contrl_pts[:,0], self.contrl_pts[:,1], self.contrl_pts[:,2], linestyle='--', marker='.', label='CPs')
        if return_axes:
            return fig, ax

def spline_curve_interp(Q, p, plot_flag=True):
    """
    Simple linear spline curve interpolation through points.
    Assumes all weights = 1.
    Q = list of points - size (ncp,2 or 3)
    p = degree of curve
    """
    ncp = Q.shape[0]
    n = ncp - 1
    m = n + p + 1
    nk = m + 1  
    # determine u_bar (spacing between interpolation points)
    temp = np.sqrt(np.linalg.norm(Q[1:,:] - Q[:-1,:], axis=1))
    # temp = np.linalg.norm(Q[1:,:] - Q[:-1,:], axis=1)
    d = np.sum(temp, axis=0)
    u_bar = np.append(0, np.cumsum(temp/d))
    u_bar[-1] = 1

    # Determine knot vector U
    U = np.zeros((nk))
    U[:p+1] = 0
    U[m-p:] = 1
    U[p+1:m-p] = [np.sum(u_bar[j:j+p])/p for j in range(1,n-p+1)]

    # Initialise a spline curve with degree and knot vector
    curve = BSplineCurve(p, U)
    curve.ndims = Q.shape[1]
    curve.weights = np.ones((ncp,1))

    # Fill coefficient matrix A using basis functions
    A = np.zeros((n+1, n+1))
    for i in range(n+1):
        span = curve.find_span(u_bar[i])
        A[span-p:span+1, i] = np.squeeze(curve.basis_funs(span, u_bar[i]))

    # Solve linear system of equations for each dimension
    if curve.ndims == 2:
        P = np.stack((np.linalg.solve(A.T, Q[:,0]),
                        np.linalg.solve(A.T, Q[:,1])), axis=1)
    elif curve.ndims == 3:
        P = np.stack((np.linalg.solve(A.T, Q[:,0]),
                        np.linalg.solve(A.T, Q[:,1]),
                        np.linalg.solve(A.T, Q[:,2])), axis=1)

    curve.contrl_pts = P
    curve.Pw = np.hstack((curve.weights * curve.contrl_pts, curve.weights))
    
    if plot_flag:
        # fig, ax = curve.plot_curve(return_axes=True)
        
        if curve.ndims == 2:
            random_ind = np.linspace(0, len(Q)-1, 101).astype(int)
            curve.plot_curve(method=2, extra_pts=Q[random_ind, :])
            # ax.scatter(Q[:,0], Q[:,1], marker='.', label='Original pts')
            # ax.legend()
    return curve

def curve_from_DVs(X, opti_options):
    DVinfo = opti_options["DVinfo"]
    p = opti_options["p"]
    ndims = opti_options["ndims"]
    # Unpack X (design variables)
    if opti_options["optCPs"]:
        CPs = np.zeros((opti_options["nCPs"], ndims))
        for i in range(ndims):
            CPs[:,i] = X[np.logical_and(DVinfo[:,0] == 1, DVinfo[:,1] == i)] * (2 - -2) + -2
    if opti_options["optWeights"]:
        weights = X[DVinfo[:,0] == 2]
    else:
        weights = opti_options["weights"]
    if opti_options["optKnots"]:
        knots = X[DVinfo[:,0] == 3]
        U = np.append(np.append([0]*(p+1), knots), [1]*(p+1))
    else:
        U = opti_options["U"]
    # Instantiate B spline curve
    curve = BSplineCurve(p, U, weights.reshape(-1,1), CPs)
    # curve.plot_curve(method=2)

    if np.any(U > 1) or np.any(curve.knots > 1):
        print(U)
        print(curve.knots)
    return curve

def wrapper(X, opti_options):

    DVinfo = opti_options["DVinfo"]
    p = opti_options["p"]
    ndims = opti_options["ndims"]
    curve = curve_from_DVs(X, opti_options)


    # Opt1: Requires projection algorithm
    # 1. Choose random selection of sample points from point list
    # 2. Project sample points onto the spline curve
    # 3. Evaluate distance (e.g. error)
    # 4. sum squares of all errors into an objective

    # Opt2: Doesn't require projection
    curve.def_mapping(npts=1000, plot_flag=False)
    # 1. Choose random selection of sample points from point list
    npts = 100
    # random_ind = np.random.random_integers(0, len(opti_options["pt_list"])-1, npts)
    random_ind = np.linspace(0, len(opti_options["pt_list"])-1, npts).astype(int)
    # 2. Obtain normalised arc length coords of these points
    s_sample = opti_options["pt_list"][random_ind, ndims]
    # 3. Map to u-coordinates on spline curve
    u_sample = curve.norms2u(s_sample)
    if np.any(u_sample > 1) :
        print(u_sample)

    # 4. Evaluate curve at these u values
    xy_sample = curve.eval_list(u_sample)
    # 5. Evaluate distance/error between evaluated points and those from point list
    dist = np.linalg.norm(opti_options["pt_list"][random_ind, 0:ndims] - xy_sample, axis=1)
    
    # Define objective
    objective = np.sqrt(np.sum(dist ** 2)) / npts
    return objective

######## Ex1
# U = [0,0,0,0,1,2,3,3,3,3]
# p = 3
# w = np.array([1,4,1,1,10,1]).reshape(-1,1)
# P = np.array([[0,0,0],[1,1,1],[1,3,2],[3,4,1],[4,5,-1],[5,7,3]])

# curve = BSplineCurve(p, U, w, P)
# curve.plot_curve(method=2)
# curve.def_mapping(npts=1000, plot_flag=True)

# u_all = np.linspace(curve.U[0], curve.U[-1], 1000)
# C1 = curve.eval_list(u_all, method=1)
# C2 = curve.eval_list(u_all, method=2)

# assert np.all(np.isclose(C1,C2))

## Check method1 and method2 produce same basis functions
# u = 2.5
# temp = np.array(curve.knots) < u
# if np.all(temp == False):
#     span = 0
# else:
#     span = np.where(curve.knots < u)[0][-1]

# s = u - curve.knots[span]
# s_vec = np.array([s**3, s**2, s, 1])
# N = np.matmul(s_vec, curve.M[:,:,span])
# N1 = curve.basis_funs(curve.find_span(u), u, ders=0)
# print(N, N1)

###### Ex2
# U = [0,0,0,1,2,3,4,4,5,5,5]
# p = 2
# u = 5/2
# curve = BSplineCurve(p, U)

# i = curve.find_span(u)
# dN = curve.basis_funs(i, u, ders=2)
# curve.eval_der(u, 2)