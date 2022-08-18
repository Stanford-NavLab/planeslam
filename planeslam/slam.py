"""SLAM functions

"""

import numpy as np

from planeslam.geometry.util import quat_to_R


def offline_slam(PCs):
    """Offline SLAM
    
    Process sequence of point clouds

    Parameters
    ----------
    PCs :

    Returns
    -------
    trajectory
    map
    
    """
