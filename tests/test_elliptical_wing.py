#%%
import matplotlib.pyplot as plt
import numpy as np
from foilpy.LL_functions import steady_LL_solve, plot_wake
from foilpy.foildef import EllipticalWing
# %matplotlib widget

def test_elliptical_wing():
    """
    This function tests that the theoretical lift coefficient of an elliptical wing
    is predicted correctly.
    """

    rho = 1.225

    # assemble elliptical wing
    span_b = 5.0
    c_root = 1.0
    u_flow = np.array([[0, -1.0, 0.1]])

    # EllipticalWing is a child class of LiftingSurface
    elliptical_wing = EllipticalWing(rt_chord=c_root,
                                     span=span_b,
                                     Re=1,
                                     nsegs=40,
                                     units='m')

    # redefine afoil polar table as idealised flat plate
    elliptical_wing.define_flat_plate_polar()

    # check plot looks OK
    elliptical_wing.plot2D()

    # assemble input dictionaries
    lifting_surface = [elliptical_wing.create_dict()]

    # do steady lifting line solve
    min_dt = c_root/np.linalg.norm(u_flow)
    out = steady_LL_solve(lifting_surface, u_flow, rho, dt=0.5, 
                            min_dt=min_dt, wake_rollup=False,
                            include_shed_vorticity=True, nit=50, delta_visc=0.0)
    u_gamma = out[0]

    # compute lift coefficient given steady vortex solve
    u_cp = u_flow + u_gamma
    lift = elliptical_wing.LL_strip_theory_forces(u_cp, rho, full_output=False)
    V = np.linalg.norm(u_cp, axis=1).reshape(-1,1)
    cl = lift / (0.5 * rho * V ** 2 * elliptical_wing.dA)

    # compute theoretical values to compare to
    alpha_theory = np.arctan(0.1/1)
    AR = span_b ** 2 /(np.pi * span_b * c_root / 4)
    cl_theory = 2*np.pi/(1 + 2/AR)*alpha_theory

    fig = plt.figure()
    plt.plot(elliptical_wing.xcp[:,0], cl, 'k-')
    plt.plot([-span_b/2, span_b/2], [cl_theory, cl_theory])
    plt.grid(True)
    plt.xlabel("Span (m)")
    plt.ylabel("Lift coefficient (-)")
    plt.ylim([0.47, 0.48])
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plot_wake(lifting_surface, out[3], out[5], ax=ax)

    assert any(np.isclose(cl, cl_theory, rtol=1e-03, atol=1e-03))


# %%
