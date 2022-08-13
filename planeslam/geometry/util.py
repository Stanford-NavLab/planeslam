"""Geometry utilities

"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from planeslam.general import normalize


def vector_projection(a, b):
    """Computes the vector projection of a onto b

    Parameters
    ----------
    a : np.array (n_dim x 1)
        Vector to project
    b : np.array (n_dim x 1)
        Vector to project onto
    
    Returns
    -------
    np.array 
        Projected vector
    
    """
    b_ = normalize(b)
    return np.dot(a, b_) * b_


def project_points_to_plane(P, n):
    """Project points onto the plane defined by normal vector

    Parameters
    ----------
    P : np.array (n_pts x n_dim)
        Set of points to project
    n : np.array (n_dim x 1)
        Plane normal vector

    Returns
    -------
    np.array (n_pts x n_dim)
        Projected set of points

    """
    V = P - np.mean(P, axis=0)
    dists = V @ n
    return P - dists @ n.T


def rot_mat_from_vecs(u, v):
    """Compute Rotation Matrix which aligns vector u with v

    https://math.stackexchange.com/a/2672702

    Parameters
    ----------
    u : np.array (n_dim x 1)
        Vector to rotate
    v : np.array (n_dim x 1)
        Vector to align with

    Returns
    -------
    np.array (n_dim x n_dim)
        Rotation matrix

    """
    # Make sure u and v are normalized
    u = normalize(u)
    v = normalize(v)
    ndim = len(u)
    w = u + v
    return 2 * w @ w.T / (w.T @ w) - np.eye(ndim)  # Rodrigues's rotation formula


def axis_angle_to_rot_mat(axis, angle):
    """Convert angle and axis of rotation pair to rotation matrix (for 3D)

    Parameters
    ----------
    angle : float
        Angle of rotation in radians
    axis : np.array (3 x 1)
        Rotation axis

    Returns
    -------
    np.array (3 x 3)
        Rotation matrix

    """
    axis = normalize(axis).flatten()
    q = np.hstack((axis * np.sin(angle/2), np.cos(angle/2)))
    return quat_to_rot_mat(q)


def quat_to_R(quat):
    """Convert quaternion to 3D rotation matrix 

    Parameters
    ----------
    quat : np.array (1 x 4)
        Quaternion in scalar-last (x, y, z, w) format

    Returns
    -------
    np.array (3 x 3)
        Rotation matrix

    """
    r = R.from_quat(quat)
    return r.as_matrix()


def R_to_quat(R_mat):
    """Convert 3D rotation matrix to quaternion
    
    Parameters
    ----------
    R : np.array (3 x 3)
        Rotation matrix

    Returns
    -------
    np.array (1 x 4)
        Quaternion in scalar-last (x, y, z, w) format

    """
    r = R.from_matrix(R_mat)
    return r.as_quat()



def skew(v):
    """Convert vector to skew symmetric matrix
    
    Parameters
    ----------
    v : np.array (3)
        Vector
    
    Returns
    -------
    np.array (3 x 3)
        Skew symmetric matrix

    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

