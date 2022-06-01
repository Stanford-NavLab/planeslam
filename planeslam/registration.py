"""Scan Registration

Functions for plane-based registration.

"""

import numpy as np

from planeslam.geometry.plane import plane_to_plane_dist
from planeslam.geometry.util import skew


def get_correspondences(source, target, norm_thresh=0.1, dist_thresh=5.0):
    """Get correspondences between two scans

    Parameters
    ----------
    source : Scan
        Source scan
    target : Scan
        target scan
    norm_thresh : float 
        Correspodence threshold for comparing normal vectors
    dist_thesh : float
        Correspondence threshold for plane to plane distance

    Returns
    -------
    correspondences : list of tuples
        List of correspondence tuples
        
    """
    P = source.planes
    Q = target.planes
    #correspondences = {k: [] for k in range(len(P))}
    correspondences = []

    for i, p in enumerate(P):
        for j, q in enumerate(Q): 
            # Check if 2 planes are approximately coplanar
            if np.linalg.norm(p.normal - q.normal) < norm_thresh:
                # Check plane to plane distance    
                if plane_to_plane_dist(p, q) < dist_thresh:
                    # Add the correspondence
                    #correspondences[i].append(j)
                    correspondences.append((i,j))
        
    return correspondences


def extract_corresponding_features(source, target, correspondences):
    """Extract corresponding normals and distances for source and target scans

    N denotes number of correspondences

    Parameters
    ----------
    source : Scan
        Source scan
    target : Scan
        Target scan

    Returns
    -------
    n_s : np.array (3N x 1)
        Stacked vector of corresponding source normals
    d_s : np.array (N x 1)
        Stacked vector of corresponding source distances
    n_t : np.array (3N x 1)
        Stacked vector of corresponding target normals
    d_t : np.array (N x 1)
        Stacked vector of corresponding target distances
    
    """
    #correspondences = get_correspondences(source, target)
    N = len(correspondences)

    P = source.planes
    Q = target.planes

    n_s = np.empty((3,N)); d_s = np.empty((N,1))
    n_t = np.empty((3,N)); d_t = np.empty((N,1)) 

    for i, c in enumerate(correspondences):
        # Source normal and distance
        n_s[:,i][:,None] = P[c[0]].normal
        d_s[i] = np.dot(P[c[0]].normal.flatten(), P[c[0]].center)
        # Target normal and distance
        n_t[:,i][:,None] = Q[c[1]].normal
        d_t[i] = np.dot(Q[c[1]].normal.flatten(), Q[c[1]].center)

    n_s = n_s.reshape((3*N,1), order='F')
    n_t = n_t.reshape((3*N,1), order='F')

    return n_s, d_s, n_t, d_t


def expmap(w):
    """Exponential map w -> R
    
    Parameters
    ----------
    w : np.array (3)
        Parameterized rotation (in so(3))

    Returns
    -------
    R : np.array (3 x 3)
        Rotation matrix (in SO(3))
    
    """
    theta = np.linalg.norm(w)

    if theta == 0:
        R = np.eye(3)
    else:
        u = w / theta
        R = np.eye(3) + np.sin(theta) * skew(u) + (1-np.cos(theta)) * np.linalg.matrix_power(skew(u), 2) 
    
    return R


def transform_normals(n, q):
    """Transform normals

    n(q) = [...,Rn_i,...]

    Parameters
    ----------
    n : np.array (3N x 1)
        Stacked vector of normals
    q : np.array (6 x 1)
        Parameterized transformation

    Returns
    -------
    np.array (3N x 1)
        Transformed normals

    """
    assert len(n) % 3 == 0, "Invalid normals vector, length should be multiple of 3"
    N = int(len(n) / 3)

    # Extract rotation matrix R from q 
    R = expmap(q[3:].flatten())

    # Apply R to n
    n = n.reshape((3, N), order='F')
    n = R @ n
    n = n.reshape((3*N, 1), order='F')
    return n


def residual(n_s, d_s, n_t, d_t, q):
    """Residual for Gauss-Newton

    Parameters
    ----------
    n_s : np.array (3N x 1)
        Stacked vector of source normals
    d_s : np.array (N x 1)
        Stacked vector of source distances
    n_t : np.array (3N x 1)
        Stacked vector of target normals
    d_t : np.array (N x 1)
        Stacked vector of target distances
    q : np.array (6 x 1)
        Parameterized transformation

    Returns
    -------
    r : np.array (4N x 1)
        Stacked vector of plane-to-plane error residuals
    n_q : np.array (3N x 1)
        Source normals transformed by q
    
    """
    n_q = transform_normals(n_s, q)

    # Transform distances
    t = q[:3]
    d_q = d_s + n_q.reshape((-1,3)) @ t 

    r = np.vstack((n_q - n_t, d_q - d_t))
    return r, n_q

# def residual(n_s, d_s, n_t, d_t, R, t):
#     """Residual for Gauss-Newton

#     Parameters
#     ----------
#     n_s : np.array (3N x 1)
#         Stacked vector of source normals
#     d_s : np.array (N x 1)
#         Stacked vector of source distances
#     n_t : np.array (3N x 1)
#         Stacked vector of target normals
#     d_t : np.array (N x 1)
#         Stacked vector of target distances
#     q : np.array (6 x 1)
#         Parameterized transformation

#     Returns
#     -------
#     r : np.array (4N x 1)
#         Stacked vector of plane-to-plane error residuals
#     n_q : np.array (3N x 1)
#         Source normals transformed by q
    
#     """
#     assert len(n_s) % 3 == 0, "Invalid normals vector, length should be multiple of 3"
#     N = int(len(n_s) / 3)

#     # Transform normals
#     n_q = n_s.reshape((3, N), order='F')
#     n_q = R @ n_q
#     n_q = n_q.reshape((3*N, 1), order='F')

#     # Transform distances
#     d_q = d_s + n_q.reshape((-1,3)) @ t 

#     r = np.vstack((n_q - n_t, d_q - d_t))
#     return r, n_q
    

def jacobian(n_s, n_q):
    """Jacobian for Gauss-Newton
    
    Parameters
    ----------
    n_s : np.array (3N x 1)
        Stacked vector of source normals
    n_q : np.array (3N x 1)
        Source normals transformed by q

    Returns
    -------
    J : np.array (4N x 6)
        Jacobian matrix of residual function with respect to q

    """
    assert len(n_s) % 3 == 0, "Invalid normals vector, length should be multiple of 3"
    N = int(len(n_s) / 3)

    J = np.empty((4*N,6))

    for i in range(N):
        Rn_i = n_q[3*i:3*i+3].flatten()
        J[4*i:4*i+3,0:3] = np.zeros((3,3))
        J[4*i:4*i+3,3:6] = skew(Rn_i)
        J[4*i+3,0:3] = -Rn_i
        J[4*i+3,3:6] = np.zeros(3)
    
    return J


def decoupled_register(source, target):
    """Register source to target scan using decoupled approach
    
    Parameters
    ----------
    source : Scan
        Source scan
    target : Scan
        Target scan

    Returns
    -------
    R_hat : np.array (3 x 3)
        Estimated rotation
    t_hat : np.array (3) 
        Estimated translation

    """
    # Find correspondences and extract features
    correspondences = get_correspondences(source, target)
    n_s, d_s, n_t, d_t = extract_corresponding_features(source, target, correspondences)

    # Estimate rotation
    H = np.zeros((3,3))
    for i in range(len(correspondences)):
        H += n_s[3*i:3*(i+1)] @ n_t[3*i:3*(i+1)].T
    u, s, v = np.linalg.svd(H)
    R_hat = u @ v

    # Estimate translation
    A = np.reshape(n_t, (-1,3))
    b = d_t - d_s
    t_hat = np.linalg.lstsq(A, b, rcond=None)[0]

    return R_hat, t_hat


def GN_register(source, target):
    """Register source to target scan using Gauss Newton approach
    
    Parameters
    ----------
    source : Scan
        Source scan
    target : Scan
        Target scan

    Returns
    -------
    R_hat : np.array (3 x 3)
        Estimated rotation
    t_hat : np.array (3) 
        Estimated translation

    """
    # Find correspondences and extract features
    correspondences = get_correspondences(source, target)
    n_s, d_s, n_t, d_t = extract_corresponding_features(source, target, correspondences)

    # Initial transformation
    t = np.array([0, 1, 0])[:,None]
    u = np.array([1, 0, 0])[:,None]
    theta = 0.1
    q = np.vstack((t, theta*u))

    # Gauss-Newton
    n_iters = 5
    lmbda = 1e-3

    for i in range(n_iters):
        r, n_q = residual(n_s, d_s, n_t, d_t, q)
        J = jacobian(n_s, n_q)
        q = q + np.linalg.inv(J.T @ J + lmbda * np.eye(6)) @ J.T @ r

    R_hat = expmap(q[3:].flatten())
    t_hat = q[:3]

    return R_hat, t_hat
