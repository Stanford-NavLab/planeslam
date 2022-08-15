"""Functions for point clouds

"""

import numpy as np

from planeslam.geometry.util import quat_to_R


def velo_preprocess(PC, pose):
    """Pre-process velodyne point cloud
    
    """
    # Shift to global frame
    R = quat_to_R(pose[3:])
    t = pose[:3]
    PC = (R @ PC.T).T + t

    # Remove points below ground plane
    PC = PC[PC[:,2] > -0.1]

    # Remove points outside of x/y room bounds
    X_BOUNDS = [-7, 7]
    Y_BOUNDS = [-3, 3]
    PC = PC[np.bitwise_and(PC[:,0] > X_BOUNDS[0], PC[:,0] < X_BOUNDS[1])]
    PC = PC[np.bitwise_and(PC[:,1] > Y_BOUNDS[0], PC[:,1] < Y_BOUNDS[1])]

    # Transform back
    PC = (R.T @ (PC - pose[:3]).T).T 

    # Remove points outside of fixed local range
    dists = np.linalg.norm(PC, axis=1)
    PC = PC[dists < 5.0]

    return PC