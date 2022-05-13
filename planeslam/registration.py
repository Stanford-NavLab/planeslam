"""Scan Registration

Functions for plane-based registration.

"""

import numpy as np

from planeslam.geometry.plane import plane_to_plane_dist


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


def extract_corresponding_features(source, target):
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
    correspondences = get_correspondences(source, target)
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