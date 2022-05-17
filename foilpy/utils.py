import numpy as np

def ms2knts(velocity):
    """Converts m/s to knots."""
    return velocity * 1.943844

def knts2ms(velocity):
    """Converts knots to m/s."""
    return velocity / 1.943844

def unit_2_meters(val, unit):
    """Converts length unit to meters."""
    if unit=='mm':
        return val/1000
    elif unit == 'cm':
        return val/100
    elif unit == 'm':
        return val

def cosspace(x_start, x_end, n_pts=100, factor=False):
    """
    COSSPACE cosine spaced vector.
       COSSPACE(X1, X2) generates a row vector of 100 cosine spaced points
       between X1 and X2.

       COSSPACE(X1, X2, n_pts) generates n_pts points between X1 and X2.

       A cosine spaced vector clusters the elements toward the endpoints:
        X1    || |  |   |    |     |     |    |   |  | ||   X2

       For negative n_pts, COSSPACE returns an inverse cosine spaced vector with
       elements sparse toward the endpoints:
         X1 |     |    |   |  | | | | | |  |   |    |     | X2

       For -2 < n_pts < 2, COSSPACE returns X2.

       COSSPACE(X1, X2, n_pts, W) clusters the elements to a lesser degree as
       dictated by W. W = 0 returns a normal cosine or arccosine spaced
       vector. W = 1 is the same as LINSPACE(X1, X2, n_pts). Experiment with W < 0
       and W > 1 for different clustering patterns.
    """
    if n_pts < 0:
        n_pts = np.floor(-n_pts)
        output = x_start + (x_end - x_start) / np.pi * np.arccos(1 - 2 * np.arange(0, n_pts) / (n_pts - 1))
    else:
        n_pts = np.floor(n_pts)
        output = x_start + (x_end - x_start) / 2 * (1 - np.cos(np.pi / (n_pts-1) * np.arange(0, n_pts)))

    if factor is not False:
        output = (1-factor) * output + factor * np.append(x_start + np.arange(0, n_pts-1) * (x_end-x_start) / (n_pts-1), x_end)

    output[0] = x_start # avoid numerical error
    output[-1] = x_end # avoid numerical error
    return output

def rotation_matrix(w, angle, deg=True):
    """
    Defines a rotation matrix about a given unit vector w and
    for a given angle.
        w is the axis to rotate about, size (3,)
        angle is the angle to rotate by in radian
    """

    if deg == True:
        angle = angle * np.pi / 180
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[w[0] ** 2 * (1 - c) + c, w[0] * w[1] * (1 - c) - w[2] * s, w[0] * w[2] * (1 - c) + w[1] * s, 0],
                  [w[1] * w[0] * (1 - c) + w[2] * s, w[1] ** 2 * (1 - c) + c, w[1] * w[2] * (1 - c) - w[0] * s, 0],
                  [w[2] * w[0] * (1 - c) - w[1] * s, w[2] * w[1] * (1 - c) + w[0] * s, w[2] ** 2 * (1 - c) + c, 0],
                  [0, 0, 0, 1]])

    return R

def apply_rotation(R, vec, dim):
    """
    Applies the given rotation matrix to the vector vec.
        R is rotation matrix size (4,4)
        vec is set of vectors (m,3), or (3,m)
        dim is the dimension that the vector is in: 0 or 1
    """
    if dim == -1:
        vec = np.vstack((vec.reshape(3,-1), np.ones((1, 1))))
        vec_rot = np.dot(R, vec)
        vec_rot = vec_rot[0:3, :].reshape(-1)
    elif dim == 0:
        vec = np.vstack((vec, np.ones((1, vec.shape[1]))))
        vec_rot = np.dot(R, vec)
        vec_rot = vec_rot[0:3, :]
    elif dim == 1:
        vec = np.hstack((vec, np.ones((vec.shape[0], 1)))).T
        vec_rot = np.dot(R, vec).T
        vec_rot = vec_rot[:, 0:3]

    return vec_rot

def translation_matrix(trans_vec):
    """
    Defines a translation matrix given a translation vector.
    """
    translation = np.array([[1, 0, 0, trans_vec[0]],
                            [0, 1, 0, trans_vec[1]],
                            [0, 0, 1, trans_vec[2]],
                            [0, 0, 0, 1]])
    return translation
