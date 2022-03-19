from source.LL_functions import eval_biot_savart, update_elmt_length
import numpy as np

def test_eval_biot_savart():
    # 1. test u_induced = 0 for any cp lying on the vortex line
    # 2. 

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

