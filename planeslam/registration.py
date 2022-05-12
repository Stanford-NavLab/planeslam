"""Scan Registration

Functions for plane-based registration.

"""

import numpy as np

from planeslam.geometry.plane import plane_to_plane_dist


def get_correspondences(P, Q, norm_thresh=0.1, dist_thresh=5.0):
    """Get correspondences between two scans

    Parameters
    ----------
    P : Scan
        First scan
    Q : Scan
        Second scan
    norm_thresh : float 
        Correspodence threshold for comparing normal vectors
    dist_thesh : float
        Correspondence threshold for plane to plane distance

    Returns
    -------
    correspondences : dict
        Dictionary storing corresponding planes in Q for each plane in P
        
    """
    P = P.planes
    Q = Q.planes
    correspondences = {k: [] for k in range(len(P))}

    for i, p in enumerate(P):
        for j, q in enumerate(Q): 
            # Check if 2 planes are approximately coplanar
            if np.linalg.norm(p.normal - q.normal) < norm_thresh:
                # Check plane to plane distance    
                if plane_to_plane_dist(p, q) < dist_thresh:
                    # Add the correspondence
                    print(i,j)
                    correspondences[i].append(j)
        
    return correspondences
