import numpy as np
from foilpy.LL_functions import eval_biot_savart, update_elmt_length

def test_eval_biot_savart():
    """
    This function tests that the biot-savart equation produces the expected outputs.
    1. test u_induced = 0 for any control point lying on the vortex line.
    """

    # generate input arrays
    xcp = np.array([[0,0,0]])
    nodes = np.array([[0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])
    gamma = np.array([1, 10, 100])

    # test code
    l0 = update_elmt_length(nodes[:,0:3], nodes[:,3:6])
    xnode1 = nodes[:,0:3].reshape(1,-1,3)
    xnode2 = nodes[:,3:6].reshape(1,-1,3)
    u_induced = eval_biot_savart(xcp, xnode1, xnode2, gamma, l0)

    # generate expected output array
    expected = np.array([[0,0,0],
                         [0,0,0],
                         [0,0,0]])

    assert np.all(u_induced == expected)

# Test auto-diff jacobian
# u_FV = np.zeros((1,3))
# f = lambda gamma: LL_residual(gamma, rho, u_BV, u_FV, u_motion, front_wing.dl, front_wing.a1, front_wing.a3, front_wing.cl_spline, front_wing.dA)
# J1 = numerical_jacobian(f, np.array(gamma_ini), 1e-4)
# print(J1)
# print(J1.shape)
# print(J-J1)
# print(np.max(J - J1))

## Test root finding algo with numerical derivative
# u_FV = np.zeros((1,3))
# f = lambda gamma: LL_residual(gamma, rho, u_BV, u_FV, u_motion, front_wing.dl, front_wing.a1, front_wing.a3, front_wing.cl_spline, front_wing.dA)
# J1 = lambda gamma: numerical_jacobian(f, np.array(gamma), 1e-4)
# gamma_root, res = newton_raphson_solver(f, J1, np.array(gamma_ini), nit=10)
