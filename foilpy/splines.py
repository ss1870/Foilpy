#%%
from this import d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from copy import deepcopy

class BSplineCurve():
    """
    B-Spline/NURBS curve class.
    Nomenclature:
        - Curve parameterised on u
        - Control points P_i
            - i = 0,...,n
            - npts in u is n+1, npts in v is m+1
        - u direction:
            - degree = p
            - knot vector = U = [[0]*(p+1), u_p+1, ..., u_r-p-1, [1]*(p+1)]
            - no knots = r+1, where r = n+p+1
    """
    def __init__(self, p, U, w=None, P=None):
        self.p = p     # Degree
        self.U = U          # Knot vector
        self.knotsU = np.array(self.U[self.p:-self.p])
        self.r = len(U) - 1 # No of knots = r + 1
        self.n = self.r - self.p - 1
        self.Np = self.n + 1
        if p == 3:
            self.M_u = self.def_m_matrix(self.knotsU, self.p)
        if np.any(P is not None):
            self.contrl_pts = P # Control points
            assert P.shape[0] - 1 == self.r - self.p - 1
            self.ndims = self.contrl_pts.shape[1]
            if np.any(w != None):
                # if weights are present then this is a NURBS curve
                self.weights = w    # Weights
            else:
                self.weights = np.ones((self.n+1,1))
            self.Pw = np.hstack((self.weights * self.contrl_pts, self.weights))
        
    def def_m_matrix(self, knots, p):
        """
        Define series of M matrices for each spline segment.
        Used for evaluating points and derivatives.
        http://and-what-happened.blogspot.com/2012/07/evaluating-b-splines-aka-basis-splines.html
        """
        if p == 3:
            nseg = len(knots) - 1
            kv = np.append(np.append([knots[0]]*2, knots), [knots[-1]]*2)

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
        else:
            raise Exception("Can only compute M matrices for order = 3.")
        return M

    def find_span(self, u, U, p, n): # A2.1
        """
        Determine the knot span index
        """
        if u > U[-1]:
            raise Exception("Query u must be less than U[end](=%d)"%U[-1])
        if u < U[0]:
            raise Exception("Query u must be greater than U[0](=%d)"%U[0])
        if u == U[n+1]:
            return n
        low = p
        high = n + 1
        mid = int((low + high) / 2)
        counter = 0
        while (u < U[mid] or u >= U[mid+1]) and counter < 1000:
            if u < U[mid]:
                high = mid
            else:
                low = mid
            mid = int((low + high) / 2)
            counter += 1
        if counter >= 1000:
            raise Exception("Find-span timed out for u = %d"%u)
        
        return mid

    def basis_funs(self, i, u, U, p, ders=0): # A2.2
        """
        Compute the non-vanishing basis functions
        """
        N = [1.0]
        left = np.zeros((p+1))
        right = np.zeros((p+1))
        calc_ders = False
        if ders > 0:
            n = ders
            calc_ders = True
        if calc_ders:
            ndu = np.zeros((p+1, p+1))
            ndu[0,0] = 1
        for j in range(1, p+1):
            left[j] = u - U[i+1-j]
            right[j] =  U[i+j] - u
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
            ders = np.zeros((n+1, p+1))
            for j in range(p+1):
                ders[0,j] = ndu[j,p]
            a = np.zeros((n+1, p+1))
            for r in range(p+1):
                s1=0
                s2=1
                a[0,0] = 1
                for k in range(1, n+1):
                    d = 0.0
                    rk = r - k
                    pk = p - k
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
                        j2 = p - r
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
            r = p
            for k in range(1,n+1):
                for j in range(p+1):
                    ders[k,j] *= r
                    r *= p - k

        if calc_ders:
            return np.array(N).reshape(-1,1), ders
        else:
            return np.array(N).reshape(-1,1)

    def one_basis_fun(self, p, i, u):
        """
        Compute Nip for a single basis function
        """
        if (i == 0 and u == self.U[0]) or (i == self.r-p-1 and u ==self.U[self.r]):
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

    def find_span2(self, knots, u):
        """
        Find span for list of knots, rather than full knot vector.
        """
        if u > knots[-1]:
            raise Exception("Query u must be less than U[end](=%d)"%knots[-1])
        if u < knots[0]:
            raise Exception("Query u must be greater than U[0](=%d)"%knots[0])
        if u == knots[-1]:
            return len(knots) - 2

        temp = knots <= u
        span = np.where(temp)[0][-1]
        return span

    def eval_curve(self, u, method=2, der1=False, der2=False):
        """
        Evaluate curve at single point u on non-dimensional arc
        """
        if (der1 or der2) and method==1:
            raise Exception("Error: derivative options only for eval method=2.")

        # Find knot span in which u resides
        span = self.find_span(u, self.U, self.p, self.n)

        if method == 1:
            N = self.basis_funs(span, u, self.U, self.p)
            Cw = np.sum(np.tile(N, (1,self.ndims+1)) * self.Pw[span-self.p:span+1,:], axis=0)
            
        elif method == 2 and self.p==3:

            s = u - self.knotsU[span-self.p]
            s_vec = np.array([s**3, s**2, s, 1])

            MCw = np.matmul(self.M_u[:,:,span-self.p], self.Pw[span-self.p:span+1,:])
            Cw = np.matmul(s_vec, MCw)

        C = Cw[:self.ndims] / Cw[self.ndims]
        if der1:
            # first derivative w.r.t u parameter
            s_vecd1 = np.array([3*s**2, 2*s, 1, 0])
            dCw1 = np.matmul(s_vecd1, MCw)
            dC1 = dCw1[:self.ndims] / Cw[self.ndims]
        if der2:
            # second derivative w.r.t u parameter
            s_vecd2 = np.array([6*s, 2, 0, 0])
            dCw2 = np.matmul(s_vecd2, MCw)
            dC2 = dCw2[:self.ndims] / Cw[self.ndims]

        if der1 and not der2:
            return C, dC1
        elif not der1 and der2:
            return C, dC2
        elif der1 and der2:
            return C, dC1, dC2
        else:
            return C

    def eval_der(self, u, d):
        """
        Alg3.2: Evaluate derivate of B spline at u.
        u = parameter location
        d = derivative (1 = 1st, 2 = 2nd)
        """
        if hasattr(self, 'weights'):
            raise Exception("Evaluating derivative not work for NURBS.")
        du = min(d, self.p)
        CK = np.zeros((du+1, 1))
        for k in range(self.p+1, d+1):
            CK[k,:] = 0
        span = self.find_span(u, self.U, self.p, self.n)
        N, nders = self.basis_funs(span, u, self.U, self.p, ders=d)
        for k in range(du+1):
            CK[k,:] = 0
            for j in range(self.p+1):
                CK[k,:] += nders[k,j] #* self.contrl_pts[span-self.p+j,:]

    def eval_list(self, u, method=2):
        """
        Evaluate curve at a list of u parameters.
        """
        if self.p != 3:
            method = 1
        try:
            C = np.zeros((len(u), self.ndims))
            for i, u in enumerate(u):
                C[i] = self.eval_curve(u, method=method)
        except:
            C = self.eval_curve(u, method=method)

        return C

    def eval_curvature(self, u):
        _, d1, d2 = self.eval_curve(u, method=2, der1=True, der2=True)
        return (d1[0] * d2[1] - d1[1] * d2[1]) / (d1[0]**2 + d1[1]**2) ** 1.5

    def def_mapping(self, npts=1000, plot_flag=False):
        # evaluate curve
        u = np.linspace(self.U[0], self.U[-1], npts)
        xy = self.eval_list(u, method=2)

        s = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))))
        self.length = s[-1]

        # define interper mappings
        self.u2s = interp1d(u, s, axis=0)
        self.u2norms = interp1d(u, s/self.length, axis=0)
        self.s2u = interp1d(s, u, axis=0)
        self.norms2u = interp1d(s/self.length, u, axis=0)
        self.norms2xy = interp1d(s/self.length, xy, axis=0)

        if plot_flag:
            fig, ax = plt.subplots()
            ax.plot(u/u[-1], s/self.length)
            ax.axis('scaled')
            ax.grid(True)
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
                N[i,j] = self.one_basis_fun(self.p, i, u)

        plt.plot(u_all, N.T)
        return N
    
    def plot_curve(self, pts=100, method=2, fig = None, ax = None,
                return_axes=False, extra_pts=None, rond=None,
                scaled=True, plotCPs=True):
        """
        Plot curve over full u domain
        """
        u_all = np.linspace(self.U[0], self.U[-1], 1000)
        C = self.eval_list(u_all, method=method)

        if fig == None:
            fig, ax = plt.subplots()
            if self.ndims == 3:
                ax = fig.add_subplot(projection='3d')
        if self.ndims == 1:

            if rond != None:
                x = rond.u2s(u_all)
                Px = rond.u2s(self.knotsU)
            else:
                x = u_all
                Px = self.knotsU
            Px = np.insert(Px, 1, (2*Px[0]+Px[1])/3)
            Px = np.insert(Px, len(Px)-1, (2*Px[-1]+Px[-2])/3)
            if np.any(extra_pts != None):
                ax.plot(extra_pts[:,0], extra_pts[:,1], linestyle='', marker='.')
            ax.plot(x, C[:,0])

            if plotCPs:
                ax.plot(Px, self.contrl_pts[:,0], linestyle='--', marker='.')

        elif self.ndims == 2:
            if np.any(extra_pts != None):
                ax.plot(extra_pts[:,0], extra_pts[:,1], linestyle='', marker='.')
            ax.plot(C[:,0], C[:,1])
            if plotCPs:
                ax.plot(self.contrl_pts[:,0], self.contrl_pts[:,1], linestyle='--', marker='.')

            if scaled:
                ax.axis('scaled')
        elif self.ndims == 3:
            ax.plot3D(C[:,0], C[:,1], C[:,2])
            if np.any(extra_pts != None):
                ax.plot3D(extra_pts[:,0], extra_pts[:,1], extra_pts[:,2], linestyle='', marker='.', color='black')
            if plotCPs:
                ax.plot3D(self.contrl_pts[:,0], self.contrl_pts[:,1], self.contrl_pts[:,2], linestyle='--', marker='.')
            if scaled:
                minval = np.amin(C)  # lowest number in the array
                maxval = np.amax(C)  # highest number in the array
                ax.set_xlim3d(minval, maxval)
                ax.set_ylim3d(minval, maxval)
                ax.set_zlim3d(minval, maxval)
        ax.grid(True)
        if return_axes:
            return fig, ax


def parameterise_curve(Q, method='centripetal', plot_flag=False):
    """
    Parameterises a set of points along a line.
    Different methods are available:
        - 'uniform'
        - 'chord'
        - 'centripetal'
        - 'Fang'
    """
    # Compute straight line vector between each point
    vec = Q[1:,:] - Q[:-1,:]
    # Compute norm (length) of each vector
    chrd_len = np.linalg.norm(vec, axis=1)

    if method == 'uniform':
        dQ = chrd_len ** 0
    elif method == 'chord':
        dQ = chrd_len ** 1
    elif method == 'centripetal':
        dQ = chrd_len ** 0.5
    elif method == 'Fang':
        triangle_chrd = np.linalg.norm(Q[2:,:] - Q[:-2,:], axis=1)
        li = np.amin(np.stack((chrd_len[:-1], chrd_len[1:], triangle_chrd), axis=1), axis=1)
        dotprod = np.sum(vec[1:,:] * vec[:-1,:], axis=1)
        thi = np.pi - np.arccos( dotprod / (chrd_len[1:] * chrd_len[:-1]))
        dQ = chrd_len ** 0.5
        dQ[:-1] += 0.1 * (0.5 * thi * li / np.sin(0.5*thi))
        dQ[1:] += 0.1 * (0.5 * thi * li / np.sin(0.5*thi))
    else:
        raise Exception("Curve parameterisation method unrecognised.")

    d = np.sum(dQ, axis=0)
    u_bar = np.append(0, np.cumsum(dQ/d))
    u_bar[-1] = 1 # avoids numerical errors

    if plot_flag:
        s = np.append(0, np.cumsum(chrd_len) / np.sum(chrd_len))
        fig, ax = plt.subplots()
        ax.plot(s, u_bar)
        ax.grid(True)
    return u_bar


def distribute_knots(u_bar, p, n_knts, Q=None, method='even_interp',
                        plot_flag=False):
    """
    Determines knot spacing for a given set of points
    """
    r = n_knts - 1
    n = r - p - 1
    nCP = n + 1

    U = np.zeros((n_knts))
    U[:p+1] = 0
    U[r-p:] = 1
    if method == 'even_interp':
        # Even spacing for spline interpolation
        U[p+1:r-p] = [np.sum(u_bar[j:j+p]) / p for j in range(1,n-p+1)]
        assert nCP == len(u_bar)
    elif method == 'even_approx':
        d = Q.shape[0] / (n - p + 1)
        for j in range(1, n-p+1):
            i = int(j*d)
            alpha = j*d - i
            U[p+j] = (1-alpha) * u_bar[i-1] + alpha * u_bar[i]
        U[-p-1:] = 1
    elif method == 'adaptive':
        Fi, fi, ui = calc_feature_func(Q, u_bar, p, n_knts, plot_flag=plot_flag)
        U = knots_from_feature(Fi, ui, u_bar, n_knts, p, plot_flag=plot_flag)

    else:
        raise Exception("Knot spacing method not recognised: " + method)

    if plot_flag:
        b=4

    return U


def calc_feature_func(Q, u_bar, p, n_knts, plot_flag=False):
    """
    Determines feature function for a given set of points Q, with 
    parameterisation u_bar. Degree is used to determine which order derivative
    to take.
    """
    m = Q.shape[0]
    # Compute derivatives up to order of approximating curve
    Qtemp = Q
    utemp = u_bar.reshape(-1,1)
    ders = {"der0" : {"q": Qtemp, "u": utemp}}
    for k in range(1, p+1):
        qk = (Qtemp[1:,:] - Qtemp[:-1,:]) / (utemp[1:,:] - utemp[:-1,:])
        uk = 0.5 * (utemp[1:,:] + utemp[:-1,:])
        ders["der%i"%(k)] = {"q": qk, "u": uk}
        Qtemp = qk
        utemp = uk

    # Calc feature function
    ui = np.append(np.append(u_bar[0], ders["der%i"%(p)]["u"][1:m-p+1]), u_bar[m-1])
    qi = ders["der%i"%(p)]["q"][1:m-p+1,:]
    fi = np.linalg.norm(qi, axis=1) ** (1/p)
    fi = np.append(np.append(0, fi), 0)

    # Cumulative feature function
    fj = 0.5 * (fi[1:] + fi[:-1] + 1e-10) * (ui[1:] - ui[:-1])
    Fi = np.append(0, np.cumsum(fj))

    # Determine deltaF based on requested number of CPs and knots
    n_iknts = n_knts - p*2
    deltaF = Fi[-1] / (n_iknts - 1)
    # Adjust cumulative integration in case of any small gaps
    fj = np.minimum(deltaF, fj)
    Fi = np.append(0, np.cumsum(fj))

    if plot_flag:
        # Plot derivatives
        fig, axes = plt.subplots(p+1,1)
        for k, ax in enumerate(axes):
            ax.plot(ders["der%i"%(k)]["u"], ders["der%i"%(k)]["q"])
        plt.title("Derivatives")

        # Plot feature function
        fig, ax = plt.subplots()
        ax.plot(ui, fi, marker='.')
        plt.title("Feature function")

        # Plot cumulative feature function and knots
        fig, ax = plt.subplots()
        ax.plot(ui, Fi)
        plt.title("Cumulative feature function")
    
    return Fi, fi, ui


def knots_from_feature(Fi, ui, u_bar, n_knts, p, plot_flag=False):
    """
    Determines a knot vector from a feature function Fi(ui).
    """
    n_iknts = n_knts - p*2 # no of internal knots
    deltaF = Fi[-1] / (n_iknts - 1)

    # Invert the feature function
    Finv = interp1d(Fi, ui)

    # Determine the knot vector
    k = np.arange(1, n_iknts+1)
    even_f_pts = (k - 1) * deltaF
    if np.isclose(even_f_pts[-1], Fi[-1]):
        even_f_pts[-1] = Fi[-1]
    Uk = Finv(even_f_pts)
    if np.isclose(Uk[0], u_bar[0]):
        Uk[0] = u_bar[0]
    if np.isclose(Uk[-1], u_bar[-1]):
        Uk[-1] = u_bar[-1]
    U = np.concatenate(([u_bar[0]]*p, Uk, [u_bar[-1]]*p))

    if plot_flag:
        # Plot cumulative feature function and knots
        fig, ax = plt.subplots()
        ax.plot(ui, Fi)
        F = interp1d(ui, Fi)
        for uk in Uk:
            ax.plot([uk,uk], [0, F(uk)], linestyle='--', color='black')
            ax.plot([0, uk], [F(uk), F(uk)], linestyle='--', color='black')
        plt.title("Cumulative feature function + knots")

    return U


def curve_approx(Q, ncp, p, u_bar=None, U=None, plot_flag=True,
                        knot_spacing='adaptive', param_method='centripetal'):
    """
    Spline curve approximation.
    Creates an approximating spline function given the points Q.
    The user chooses the number of control points (ncp) and degree (p).
    The user may also specify the parameterisation of the points (u_bar),
    otherwise the chordwise method is used. The user may also specify the knot
    vector, otherwise two methods are available:
        knot_spacing='even_approx'  : Even spacing between knots.
        knot_spacing='adaptive'     : Adaptive knot spacing method based on Fast Automatic Knot Placement Method for Accurate B-spline Curve Fitting (https://doi.org/10.1016/j.cad.2020.102905)
    'adaptive' is recommended unless the data is noisy.
    """
    npts = Q.shape[0]
    m = npts
    n = ncp - 1
    n_knts = n + p + 2
    if ncp + p + 1 < 8:
        raise Exception("No of control points for approximation must be >= (7 - degree)")

    # Get u_bar from spacing between supplied points
    if np.all(u_bar == None):
        u_bar = parameterise_curve(Q, method=param_method)

    # Determine knot vector (if not provided)
    if np.all(U == None):
        if np.all(np.isclose(np.diff(Q, axis=0) , 0)):
            knot_spacing = 'even_approx'
        U = distribute_knots(u_bar, p, n_knts, Q=Q, method=knot_spacing, 
                                plot_flag=plot_flag)

    # Initialise a spline curve with degree and knot vector
    curve = BSplineCurve(p, U)
    curve.ndims = Q.shape[1]
    curve.weights = np.ones((ncp,1))

    if np.all(np.isclose(np.diff(Q, axis=0) , 0)):
        P = Q[0] * np.ones((ncp,1))
    else:
        # Fill coefficient matrix N using basis functions
        Nall = np.zeros((m, n+1))
        for k in range(0, m):
            span = curve.find_span(u_bar[k], curve.U, p, n)
            Nall[k, span-p:span+1] = np.squeeze(curve.basis_funs(span, u_bar[k], U, p))

        N = Nall[1:m-1, 1:n]

        # Fill Rk vector
        Rk = Q - Nall[:,0].reshape(-1,1) * Q[0,:].reshape(1,-1) - Nall[:,-1].reshape(-1,1) * Q[-1,:].reshape(1,-1)

        # Define R vector
        R = np.matmul(N.T, Rk[1:-1,:])

        NtN = np.matmul(N.T, N)

        # Solve linear system of equations for each dimension
        P = np.zeros((ncp-2, curve.ndims))
        for i in range(curve.ndims):
            P[:,i] = np.linalg.solve(NtN, R[:,i])

        # Add end points
        P = np.append(
                np.append(Q[0,:].reshape(1,-1), P, axis=0),
                Q[-1,:].reshape(1,-1), axis=0)

    curve.contrl_pts = P
    curve.Pw = np.hstack((curve.weights * curve.contrl_pts, curve.weights))

    if plot_flag:
        Qnew = deepcopy(Q)
        if curve.ndims == 1:
            Qnew = np.hstack((u_bar.reshape(-1,1), Q.reshape(-1,1)))
        fig, ax = curve.plot_curve(method=2, scaled=False, return_axes=True)
        # Compute diff between eval pts and equivalent curve points
        diff = np.linalg.norm(curve.eval_list(u_bar) - Q, axis=1)
        # Plot scatter of points with color to indicate distance
        diffplot = ax.scatter(Qnew[:,0],
                        Qnew[:,1],
                        Qnew[:,2],
                        c=diff, marker='.')
        plt.colorbar(diffplot)

        Q_rng = np.max((np.max(Q, axis=0) - np.min(Q, axis=0))) + 1e-12
        # err_max = 1 / Q_rng * np.max(np.linalg.norm(curve.eval_list(u_bar) - Q))
        # err_rms = 1 / Q_rng * np.sqrt(np.sum(np.linalg.norm(curve.eval_list(u_bar) - Q) ** 2) / m)
        err_max = np.max(diff)
        err_rms = np.sqrt(np.sum(diff ** 2) / m)
        plt.title("Max error=%f, RMS error=%f"%(err_max, err_rms))

    
    return curve


def curve_interp(Q, p, u_bar=None, U=None, param_method='centripetal',
                    plot_flag=True):
    """
    Simple linear spline curve interpolation through points.
    Assumes all weights = 1.
    Q = list of points - size (ncp, 2 or 3)
    p = degree of curve
    """
    ncp = Q.shape[0]
    n = ncp - 1
    r = n + p + 1
    n_knts = r + 1

    if np.all(u_bar == None):
        # Determine u_bar
        u_bar = parameterise_curve(Q, method=param_method)

    if np.all(U == None):
        # Determine knot vector U
        U = distribute_knots(u_bar, p, n_knts, method='even_interp', plot_flag=False)

    # Initialise a spline curve with degree and knot vector
    curve = BSplineCurve(p, U)
    curve.ndims = Q.shape[1]
    curve.weights = np.ones((ncp,1))

    # Fill coefficient matrix A using basis functions
    A = np.zeros((n+1, n+1))
    for i in range(n+1):
        span = curve.find_span(u_bar[i], U, p, n)
        A[span-p:span+1, i] = np.squeeze(curve.basis_funs(span, u_bar[i], U, p))

    # Solve linear system of equations for each dimension
    P = np.zeros((ncp, curve.ndims))
    for i in range(curve.ndims):
        P[:,i] = np.linalg.solve(A.T, Q[:,i])

    curve.contrl_pts = P
    curve.Pw = np.hstack((curve.weights * curve.contrl_pts, curve.weights))
    
    if plot_flag:
        if curve.ndims == 1:
            Qplot = np.hstack((u_bar.reshape(-1,1), Q))
        else:
            Qplot = deepcopy(Q)

        # if curve.ndims == 2:
        # random_ind = np.linspace(0, len(Q)-1, 101).astype(int)
        curve.plot_curve(method=2, extra_pts=Qplot)
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
# u = 2
# span = curve.find_span2(curve.knotsU, u)
# s = u - curve.knotsU[span]
# s_vec = np.array([s**3, s**2, s, 1])
# N = np.matmul(s_vec, curve.M_u[:,:,span])
# N1 = curve.basis_funs(curve.find_span(u, U, p, curve.n), u, U, p, ders=0)
# print(N, N1)

## Check time for find span and findspan2
# import timeit
# num_runs = 10000
# u = 0.1
# p = 3
# U = np.concatenate(([0]*p, np.linspace(0,1,100), [1]*p))
# curve = BSplineCurve(p, U)
# func = lambda : curve.find_span(u,U,p,curve.n)
# func1 = lambda : curve.find_span2(curve.knotsU, u)

# duration = timeit.Timer(func).timeit(number = num_runs)
# print(f'On average it took {duration/num_runs} seconds')
# duration = timeit.Timer(func1).timeit(number = num_runs)
# print(f'On average it took {duration/num_runs} seconds')

###### Ex2
# U = [0,0,0,1,2,3,4,4,5,5,5]
# p = 2
# u = 5/2
# curve = BSplineCurve(p, U)

# i = curve.find_span(u, U, p, n)
# dN = curve.basis_funs(i, u, U, p, ders=2)
# curve.eval_der(u, 2)


##### Opti trials:
# from scipy.optimize import Bounds, LinearConstraint, minimize
# from scipy.optimize import SR1
# X0 = twist_curv.knots[1:-1]
# nDVs = len(X0)
# LB = np.zeros((nDVs))
# UB = np.ones((nDVs))
# F = lambda X : wrapper(X, Q, ncp, p, rond0)
# F(X0)

# lin_con_coeffs = np.zeros((nDVs,nDVs))
# lin_con_lb = -np.inf * np.ones((nDVs))
# lin_con_ub = np.inf * np.ones((nDVs))
# knt_counter = 0
# cons = []
# for i in range(nDVs):
#     lin_con_coeffs[i,i] = 1
#     if knt_counter < (nDVs-1):
#         lin_con_coeffs[i,i+1] = -1
#         lin_con_ub[i] = 0
#         temp = lambda x: x[i+1] - x[i]
#     else:
#         lin_con_ub[i] = 1
#         temp = lambda x: 1 - x[i] 
#     cons.append({'type': 'ineq', 'fun': temp})
#     knt_counter += 1
# bounds = Bounds(LB, UB)
# lin_constr = LinearConstraint(lin_con_coeffs, lin_con_lb, lin_con_ub)


# # jac="2-point", hess=SR1(),
# res = minimize(F, X0, method='trust-constr', jac="3-point",
#                constraints=[lin_constr],
#                options={'verbose': 1, 'maxiter': 50, 'disp': True}, 
#                bounds=bounds)
# wrapper(res.x, Q, ncp, p, rond0, plot_flag=True)


# from scipy.optimize import shgo
# res = shgo(F, np.stack((LB,UB), axis=1), iters=4, constraints=cons,
#                options={'disp': True})



# def wrapper(X, Q, ncp, p, rond0, plot_flag=False):
#     U = np.append(np.append([0]*(p+1), X), [1]*(p+1))
#     try: 
#         curv, u_bar = spl.spline_curve_approx(Q, ncp, p, U=U, 
#                     gombocX=True, rond0=rond0, plot_flag=plot_flag)
        
#         obj = np.sum(np.linalg.norm(Q - curv.eval_list(u_bar), axis=1) ** 2)
#     except:
#         obj = 100
#     return obj

