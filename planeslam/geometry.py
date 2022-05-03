"""Geometry utilities

"""

import numpy as np


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
    b_ = b / np.linalg.norm(b)
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
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    ndim = len(u)
    w = u + v
    return 2 * w @ w.T / (w.T @ w) - np.eye(ndim)  # Rodrigues's rotation formula


def plane_to_plane_dist(plane_1, normal_1, plane_2, normal_2):
    """Plane-to-Plane distance
    
    Shortest distance between two (rectangularly bounded) planes. Computed
    by taking the centroid to centroid vector, and projecting it along the
    average normal of the two planes, and taking the norm of the projected
    vector. Meant for planes with close normals.

    Parameters
    ----------
    plane_1 : np.array (4 x 3)
        Rectangularly bounded plane represented by 4 vertices
    plane_2 : np.array (4 x 3)
        Rectangularly bounded plane represented by 4 vertices
    
    Returns
    -------
    float
        Plane-to-plane distance
    
    """
    centroid_1 = np.mean(plane_1, axis=0)
    centroid_2 = np.mean(plane_2, axis=0)
    c2c_vector = centroid_1 - centroid_2  # NOTE: may be issue with c2c_vector pointing opposite to avg_normal
    avg_normal = (normal_1 + normal_2) / 2
    return np.linalg.norm(vector_projection(c2c_vector, avg_normal))

