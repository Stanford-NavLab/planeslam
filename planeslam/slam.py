"""SLAM functions

"""

import numpy as np
from graphslam.graph import Graph
from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se3 import PoseSE3

from planeslam.scan import pc_to_scan
from planeslam.geometry.util import quat_to_R
from planeslam.registration import robust_GN_register


def offline_slam(PCs, init_pose):
    """Offline SLAM
    
    Process sequence of point clouds to generate a trajectory  
    estimate and map.

    Parameters
    ----------
    PCs : list of np.array 
        List of point clouds

    Returns
    -------
    trajectory : 
        Sequence of poses
    map : Scan
        Final map composed from merged scans
    
    """
    # For airsim
    N = len(PCs)

    # Relative transformations
    R_hats = []
    t_hats = []

    # Absolute poses
    R_abs, t_abs = init_pose
    poses = N * [None]
    poses[0] = (R_abs, t_abs)

    # Scans
    scans = N * [None]
    scans[0] = pc_to_scan(PCs[0])

    # Pose graph

    for i in range(1, N):
        P = PCs[i]
        
        # Extract scan
        scans[i] = pc_to_scan(P)
        scans[i].remove_small_planes(area_thresh=5.0)

        # Registration
        R_hat, t_hat = robust_GN_register(scans[i], scans[i-1])
        t_abs += (R_abs @ t_hat).flatten()
        R_abs = R_hat @ R_abs
        poses[i] = (R_abs, t_abs)



