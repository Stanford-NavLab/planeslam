"""SLAM functions

"""

import numpy as np

from planeslam.scan import pc_to_scan
from planeslam.geometry.util import quat_to_R


def offline_slam(PCs):
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
    N = len(PCs)

    for i in range(N):
        P = PCs[i]
        # 
