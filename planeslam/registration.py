"""Scan Registration

Functions for plane-based registration.

"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from planeslam.geometry.plane import plane_to_plane_dist
from planeslam.geometry.util import skew
from planeslam.geometry.rectangle import Rectangle


# def get_correspondences(source, target, norm_thresh=0.3, dist_thresh=5.0):
#     """Get correspondences between two scans

#     Parameters
#     ----------
#     source : Scan
#         Source scan
#     target : Scan
#         target scan
#     norm_thresh : float 
#         Correspodence threshold for comparing normal vectors
#     dist_thesh : float
#         Correspondence threshold for plane to plane distance

#     Returns
#     -------
#     correspondences : list of tuples
#         List of correspondence tuples
        
#     """
#     P = source.planes
#     Q = target.planes
#     correspondences = []

#     for i, p in enumerate(P):
#         # Compute projection of p onto it's own basis
#         p_proj = (np.linalg.inv(p.basis) @ p.vertices.T).T
#         p_rect = Rectangle(p_proj[:,0:2])

#         for j, q in enumerate(Q):
#             # Check if 2 planes are approximately coplanar
#             if np.linalg.norm(p.normal - q.normal) < norm_thresh:
#                 # Check plane to plane distance    
#                 if plane_to_plane_dist(p, q) < dist_thresh:
#                     # Project q onto p's basis
#                     q_proj = (np.linalg.inv(p.basis) @ q.vertices.T).T
#                     # Check overlap
#                     q_rect = Rectangle(q_proj[:,0:2])
#                     if p_rect.is_intersecting(q_rect):    
#                         # Add the correspondence
#                         correspondences.append((i,j))
        
#     return correspondences


def get_correspondences(source, target):
    """Get correspondences between two scans

    """
    # Determine an initial set of matches based on heuristic scoring
    n = len(source.planes) # source P
    m = len(target.planes) # target Q
    score_mat = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            n1 = source.planes[i].normal
            n2 = target.planes[j].normal
            c1 = source.planes[i].center
            c2 = target.planes[j].center
            a1 = source.planes[i].area()
            a2 = target.planes[j].area()
            #score_mat[i,j] = 100 * np.linalg.norm(n1 - n2) + np.linalg.norm(c1 - c2) + 0.1 * np.abs(a1 - a2)
            score_mat[i,j] = 20 * np.linalg.norm(n1 - n2) + np.linalg.norm(c1 - c2)

    #matches = linear_sum_assignment(score_mat)
    
    # Prune the matches based on threshold requirements
    matches = np.argmin(score_mat, axis=1)
    corrs = []
    target_corresponded = []  # only allow each target plane to be corresponded once
    #for i in range(len(matches[0])):
    for i, j in enumerate(matches):
        # n1 = source.planes[matches[0][i]].normal.flatten()
        # n2 = target.planes[matches[1][i]].normal.flatten()
        # c1 = source.planes[matches[0][i]].center
        # c2 = target.planes[matches[1][i]].center
        n1 = source.planes[i].normal.flatten()
        n2 = target.planes[j].normal.flatten()
        c1 = source.planes[i].center
        c2 = target.planes[j].center
        #if np.dot(n1, n2) > 0.707 and plane_to_plane_dist(source.planes[i], target.planes[j]) < 5.0:  # 45 degrees
        if np.dot(n1, n2) > 0.707 and np.linalg.norm(c1 - c2) < 20.0:
            if j not in target_corresponded:
                corrs.append((i,j))
                target_corresponded.append(j)
    
    return corrs


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


def so3_expmap(w):
    """SO(3) exponential map w -> R
    
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


def se3_expmap(v):
    """SE(3) exponential map v -> T
    
    Parameters
    ----------
    v : np.array (3)
        Parameterized rotation (in so(3))

    Returns
    -------
    R : np.array (3 x 3)
        Rotation matrix (in SO(3))
    
    """
    t = v[:3]
    w = v[3:]
    theta = np.linalg.norm(w)

    R = so3_expmap(w) 

    V = np.eye(3) + ((1-np.cos(theta))/theta**2) * skew(w) + ((theta-np.sin(theta))/theta**3) * np.linalg.matrix_power(skew(w), 2)
    
    return np.vstack((np.hstack((R, (V@t)[:,None])), np.array([[0, 0, 0, 1]])))


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
    R = so3_expmap(q[3:].flatten())

    # Apply R to n
    n = n.reshape((3, N), order='F')
    n = R @ n
    n = n.reshape((3*N, 1), order='F')
    return n


def residual(n_s, d_s, n_t, d_t, T):
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
    T : np.array (6 x 1)
        Transformation matrix

    Returns
    -------
    r : np.array (4N x 1)
        Stacked vector of plane-to-plane error residuals
    n_q : np.array (3N x 1)
        Source normals transformed by q
    
    """
    assert len(n_s) % 3 == 0, "Invalid normals vector, length should be multiple of 3"
    N = int(len(n_s) / 3)
    R = T[:3,:3]
    t = T[:3,3][:,None]

    # Transform normals
    n_q = n_s.reshape((3, N), order='F')
    n_q = R @ n_q
    n_q = n_q.reshape((3*N, 1), order='F')

    # Transform distances
    d_q = d_s + n_q.reshape((-1,3)) @ t 

    r = np.vstack((n_q - n_t, d_q - d_t))
    return r, n_q
    

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
        J[4*i:4*i+3,3:6] = -skew(Rn_i)
        J[4*i+3,0:3] = Rn_i
        J[4*i+3,3:6] = np.zeros(3)
    
    return J


def solve_translation(R, n_s, d_s, d_t):
    """Given rotation, solve for translation using least-squares

    Parameters
    ----------
    R : np.array (3 x 3)
        Estimated rotation
    n_s : np.array (3N x 1)
        Stacked vector of source normals
    d_s : np.array (N x 1)
        Stacked vector of source distances
    d_t : np.array (N x 1)
        Stacked vector of target distances

    Returns
    -------
    t_hat : np.array (3 x 1)
        Estimated translation
    
    """
    Rn_s = (R @ n_s.reshape((3, -1), order='F'))
    t_hat = np.linalg.lstsq(Rn_s.T, d_t - d_s, rcond=None)[0]
    t_res = np.abs(Rn_s.T @ t_hat - (d_t - d_s))
    t_loss = np.linalg.norm(t_res)**2
    #print("final translation loss: ", np.linalg.norm(Rn_s.T @ t_hat - (d_t - d_s))**2)
    #print("translation residuals: ", Rn_s.T @ t_hat - (d_t - d_s))
    return t_hat, t_res, t_loss


def solve_rotation_SVD(n_s, n_t):
    """
    
    """
    H = np.zeros((3,3))
    for i in range(int(len(n_s)/3)):
        H += n_s[3*i:3*(i+1)] @ n_t[3*i:3*(i+1)].T
    u, s, v = np.linalg.svd(H)
    R_hat = u @ v
    return R_hat


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
    t_hat : np.array (3 x 1) 
        Estimated translation

    """
    # Find correspondences and extract features
    correspondences = get_correspondences(source, target)
    n_s, d_s, n_t, d_t = extract_corresponding_features(source, target, correspondences)

    # Estimate rotation
    R_hat = solve_rotation_SVD(n_s, n_t)

    # Estimate translation
    t_hat = solve_translation(R_hat, n_s, d_s, d_t)[0]

    return R_hat, t_hat


def decoupled_opt(source, target, correspondences):
    """
    
    """
    n_s, d_s, n_t, d_t = extract_corresponding_features(source, target, correspondences)

    # Estimate rotation
    H = np.zeros((3,3))
    for i in range(len(correspondences)):
        H += n_s[3*i:3*(i+1)] @ n_t[3*i:3*(i+1)].T
    u, s, v = np.linalg.svd(H)
    R_hat = u @ v

    # Estimate translation
    t_hat, t_res, t_loss = solve_translation(R_hat, n_s, d_s, d_t)

    return R_hat, t_hat, t_loss, t_res


def robust_decoupled_register(source, target):
    """
    
    """
    # Find correspondences and extract features
    correspondences = get_correspondences(source, target)
    
    # Do registration
    R_hat, t_hat, t_loss, t_res = decoupled_opt(source, target, correspondences)

    max_faults = 3
    num_faults = 0

    # Check translation loss
    while t_loss > 1.0 and num_faults < max_faults:
        fault = np.argmax(t_res)
        del correspondences[fault]
        # Redo registration
        #print("re-running registration")
        R_hat, t_hat, t_loss, t_res = decoupled_opt(source, target, correspondences)
        num_faults += 1

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
    t_hat : np.array (3 x 1) 
        Estimated translation

    """
    # Find correspondences and extract features
    correspondences = get_correspondences(source, target)
    n_s, d_s, n_t, d_t = extract_corresponding_features(source, target, correspondences)

    # Initial transformation
    T = np.eye(4)

    # Gauss-Newton
    n_iters = 20
    lmbda = 0.0
    mu = 1.0

    for i in range(n_iters):
        r, n_q = residual(n_s, d_s, n_t, d_t, T)
        print("loss: ", np.linalg.norm(r)**2)
        J = jacobian(n_s, n_q)
        dv = -mu * np.linalg.inv(J.T @ J + lmbda*np.eye(6)) @ J.T @ r
        T = se3_expmap(dv.flatten()) @ T
    
    r, _ = residual(n_s, d_s, n_t, d_t, T)
    print("final loss: ", np.linalg.norm(r)**2)

    R_hat = T[:3,:3]
    t_hat = T[:3,3]

    return R_hat, t_hat


def so3_residual(R, n_s, n_t):
    """SO(3) Residual

    Residual for rotation only SO(3) registration

    Parameters
    ----------
    R : np.array (3 x 3)
        Current rotation estimate
    n_s : np.array (3N x 1)
        Source normal vectors
    n_t : np.array (3N x 1)
        Target normal vectors

    Returns
    -------
    r : np.array (3N x 1)
        Residual vector
    n_q : np.array (3N x 1)
        Transformed source normals
    
    """
    n_q = (R @ n_s.reshape((3, -1), order='F')).reshape((-1, 1), order='F')
    r = n_q - n_t
    return r, n_q


def so3_jacobian(n_q):
    """ SO(3) Jacobian

    Jacobian for rotation only SO(3) registration

    Parameters
    ----------
    n_q : np.array (3N x 1)
        Transformed source normals

    Returns
    -------
    J : np.array (3N x 3)
        Jacobian matrix
    
    """
    N = int(len(n_q) / 3)
    J = np.empty((len(n_q), 3))

    for i in range(N):
        Rn_i =  n_q[3*i:3*i+3].flatten()
        J[3*i:3*i+3,:] = -skew(Rn_i)

    return J


def solve_rotation_GN(n_s, n_t):
    """
    
    """
    R_hat = np.eye(3)

    n_iters = 5
    lmbda = 1e-8
    mu = 1.0

    for i in range(n_iters):
        r, n_q = so3_residual(R_hat, n_s, n_t)
        #print("  loss: ", np.linalg.norm(r)**2)
        J = so3_jacobian(n_q)
        dw = - mu * np.linalg.inv(J.T @ J + lmbda*np.eye(3)) @ J.T @ r
        R_hat = so3_expmap(dw.flatten()) @ R_hat
    
    return R_hat


def decoupled_GN_register(source, target):
    """Decoupled Gauss Newton

    Register source to target scan by first estimating rotation only using Gauss Newton,
    then solving for translation.
    
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
    t_hat : np.array (3 x 1) 
        Estimated translation

    """
    # Find correspondences and extract features
    correspondences = get_correspondences(source, target)
    n_s, d_s, n_t, d_t = extract_corresponding_features(source, target, correspondences)

    # Rotation estimation
    R_hat = solve_rotation_GN(n_s, n_t)
    
    r, _ = so3_residual(R_hat, n_s, n_t)
    print("final rotation loss: ", np.linalg.norm(r)**2)

    # Translation estimation
    t_hat = solve_translation(R_hat, n_s, d_s, d_t)

    return R_hat, t_hat


def decoupled_GN_opt(source, target, correspondences):
    """
    
    """
    n_s, d_s, n_t, d_t = extract_corresponding_features(source, target, correspondences)

    # Rotation estimation
    R_hat = solve_rotation_GN(n_s, n_t)
    
    r, _ = so3_residual(R_hat, n_s, n_t)
    #print(" final rotation loss: ", np.linalg.norm(r)**2)

    # Translation estimation
    t_hat, t_res, t_loss = solve_translation(R_hat, n_s, d_s, d_t)

    return R_hat, t_hat, t_loss, t_res


def robust_GN_register(source, target, t_loss_thresh=1.0, max_faults=3):
    """Robust (decoupled) Gauss-newton
    
    """
    # Find correspondences and extract features
    correspondences = get_correspondences(source, target)
    
    # Do registration
    R_hat, t_hat, t_loss, t_res = decoupled_GN_opt(source, target, correspondences)

    num_faults = 0

    # Check translation loss
    while t_loss > t_loss_thresh and num_faults < max_faults:
        fault = np.argmax(t_res)
        print("deleting ", correspondences[fault])
        del correspondences[fault]
        # Redo registration
        #print("re-running registration")
        R_hat, t_hat, t_loss, t_res = decoupled_GN_opt(source, target, correspondences)
        num_faults += 1

    return R_hat, t_hat


def solve_rotation_basis(source, target):
    """Solve for rotation between two scans with their bases
    
    """
    return target.basis @ source.basis.T


def decoupled_basis_opt(source, target, correspondences):
    """
    
    """
    n_s, d_s, n_t, d_t = extract_corresponding_features(source, target, correspondences)

    # Rotation estimation
    R_hat = solve_rotation_basis(source, target)
    
    r, _ = so3_residual(R_hat, n_s, n_t)
    #print(" final rotation loss: ", np.linalg.norm(r)**2)

    # Translation estimation
    t_hat, t_res, t_loss = solve_translation(R_hat, n_s, d_s, d_t)

    return R_hat, t_hat, t_loss, t_res


def robust_basis_register(source, target):
    """
    
    """
    # Find correspondences and extract features
    correspondences = get_correspondences(source, target)
    
    # Do registration
    R_hat, t_hat, t_loss, t_res = decoupled_basis_opt(source, target, correspondences)

    max_faults = 3
    num_faults = 0

    # Check translation loss
    while t_loss > 1.0 and num_faults < max_faults:
        fault = np.argmax(t_res)
        del correspondences[fault]
        # Redo registration
        #print("re-running registration")
        R_hat, t_hat, t_loss, t_res = decoupled_basis_opt(source, target, correspondences)
        num_faults += 1

    return R_hat, t_hat


def iterative_register(source, target):
    """Iterative register

    Perform multiple iterations of 
        1. find correspondences
        2. optimize
        3. transform planes
    
    """
    

def loop_closure_register(source, target, source_pose, target_pose, t_loss_thresh=1.0, max_faults=3):
    """Register two scans for loop closure

    Use initial poses to get correspondences

    Parameters
    ----------
    source : Scan
        Source scan
    target : Scan
        Target scan
    source_pose : tuple
        Tuple (R,t) of source pose
    target_pose : tuple
        Tuple (R,t) of source pose

    """
    source_transformed = deepcopy(source)
    source_transformed.transform(source_pose[0], source_pose[1])
    target_transformed = deepcopy(target)
    target_transformed.transform(target_pose[0], target_pose[1])

    correspondences = get_correspondences(source_transformed, target_transformed)

    # Do registration
    R_hat, t_hat, t_loss, t_res = decoupled_GN_opt(source, target, correspondences)

    num_faults = 0

    # Check translation loss
    while t_loss > t_loss_thresh and num_faults < max_faults:
        fault = np.argmax(t_res)
        print("deleting ", correspondences[fault])
        del correspondences[fault]
        # Redo registration
        #print("re-running registration")
        R_hat, t_hat, t_loss, t_res = decoupled_GN_opt(source, target, correspondences)
        num_faults += 1

    return R_hat, t_hat