"""Scan Registration

"""

import numpy as np

from planeslam.general import downsample


def correspondences(P, Q):
    """Get correspondences between two scans

    Parameters
    ----------
    P : Scan
        First scan
    Q : Scan
        Second scan

    Returns
    -------
    correspondences :
        
    """
    

