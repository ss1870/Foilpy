#%%
import numpy as np
from foilpy.splines import BSplineCurve

def test_spline_derivatives():
    """
    This function tests spline analytic derivatives match numerical derivatives.
    """
    # Define cubic spline curve
    U = [0,0,0,0,1,2,3,3,3,3]
    p = 3
    P = np.array([[0,0,0],[1,1,1],[1,3,2],[3,4,1],[4,5,-1],[5,7,3]])
    w = np.ones((P.shape[0],1))
    curve = BSplineCurve(p, U, w, P)

    # Finite difference step size
    h = 1e-8

    # Loop through a range of u values and compare analytic vs numerical derivatives
    u = np.linspace(h, 3-h, 10)
    for ui in u:
        C, der1, der2 = curve.eval_curve(ui, method=2, der1=True, der2=True)

        Cmh = curve.eval_curve(ui-h)
        Cph = curve.eval_curve(ui+h)
        der1_FD = (Cph - Cmh)/2/h
        assert np.all(np.isclose(der1_FD, der1, rtol=1e-06, atol=1e-06))

        _, d1mh = curve.eval_curve(ui-h, method=2, der1=True)
        _, d1ph = curve.eval_curve(ui+h, method=2, der1=True)
        der2_FD = (d1ph - d1mh) / 2 / h
        assert np.all(np.isclose(der2_FD, der2, rtol=1e-06, atol=1e-06))

