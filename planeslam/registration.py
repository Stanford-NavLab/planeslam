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
