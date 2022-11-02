#%%
import numpy as np
import foilpy.splines as spl
# import matplotlib.pyplot as plt
# import importlib
# importlib.reload(spl)
# %matplotlib widget

def test_spline_derivatives():
    """
    This function tests spline analytic derivatives match numerical derivatives.
    """
    # Define cubic spline curve
    U = [0,0,0,0,1,2,3,3,3,3]
    p = 3
    P = np.array([[0,0,0],[1,1,1],[1,3,2],[3,4,1],[4,5,-1],[5,7,3]])
    w = np.ones((P.shape[0],1))
    curve = spl.BSplineCurve(p, U, w, P)

    # Finite difference step size
    h = 1e-8

    # Loop through a range of u values and compare analytic vs numerical derivatives
    u = np.linspace(h, 3-h, 10)
    for ui in u:
        # Derivatives of basis functions - note this method only works for p=3
        span = curve.find_span(ui, U, p, curve.n)
        dN1 = curve.basis_deriv(span, ui)

        Nmh = curve.basis_funs(span, ui-h, U, p, ders=0)
        Nph = curve.basis_funs(span, ui+h, U, p, ders=0)
        dN1_FD = (Nph - Nmh) / 2 / h

        assert np.all(np.isclose(dN1_FD, dN1.reshape(-1,1), rtol=1e-06, atol=1e-06))

        # Derivatives of curve coordinates
        C, der1, der2 = curve.eval_curve(ui, method=2, der1=True, der2=True)

        Cmh = curve.eval_curve(ui-h)
        Cph = curve.eval_curve(ui+h)
        der1_FD = (Cph - Cmh)/2/h
        assert np.all(np.isclose(der1_FD, der1, rtol=1e-06, atol=1e-06))

        _, d1mh = curve.eval_curve(ui-h, method=2, der1=True)
        _, d1ph = curve.eval_curve(ui+h, method=2, der1=True)
        der2_FD = (d1ph - d1mh) / 2 / h
        assert np.all(np.isclose(der2_FD, der2, rtol=1e-06, atol=1e-06))

def test_constrained_curve_approx():

    # Define set of points from a sine curve
    nPtsApprox = 30
    x = np.linspace(0, 2*np.pi, nPtsApprox)
    y = x * np.sin(2*x - x)
    Q = np.stack((x,y), axis=1)

    # Define constrained approximation
    p = 3       # degree
    ncp = 8     # no control points
    Wq = np.ones((Q.shape[0],1))    # weightings
    Wq[0] = -1
    Wq[-1] = -1
    curve = spl.constrained_approx(Q, Wq, ncp, p, 
                u_bar=None, U=None, plot_flag=False,
                knot_spacing='adaptive', param_method='centripetal')

    # Test points are constrained correctly
    assert (np.all(np.isclose(curve.eval_curve(0), Q[0,:])))
    assert (np.all(np.isclose(curve.eval_curve(1), Q[-1,:])))

    # Do another approximation with constrained derivatives
    D = np.array([[0,5], [10,0]])
    s = 1
    I = np.array([0, nPtsApprox-1])
    Wd = np.array([-1, -1])
    curve = spl.constrained_approx(Q, Wq, ncp, p,
                D=D, s=s, I=I, Wd=Wd,
                u_bar=None, U=None, plot_flag=False, plot_extra=False,
                knot_spacing='adaptive', param_method='centripetal')

    C, dC1 = curve.eval_curve(0, der1=True)
    C1, dC11 = curve.eval_curve(1, der1=True)

    # Test derivatives are constrained correctly
    assert (np.all(np.isclose(dC1, D[0,:])))
    assert (np.all(np.isclose(dC11, D[-1,:])))
