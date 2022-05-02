"""General utilities

"""

import numpy as np


def NED_to_ENU(P):
    """Convert a set of 3D points from NED to ENU

    Parameters
    ----------
    P : np.array (n_pts x 3)
        NED points to convert

    Returns
    -------
    np.array (n_pts x 3)
        points in ENU

    """
    P = P[:,[1,0,2]]  # Swap x and y
    P[:,2] = -P[:,2]  # Negate z
    return P


def remap(arr, k, v):
    """Remap the elements of an array using the map k->v

    https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python

    Elements of arr which do not appear in k will be unchanged.

    Parameters
    ----------
    arr : np.array 
        Array to remap
    k : np.array (1 x n_vals)
        Values to replace (i.e. "keys")
    v : np.array (1 x n_vals)
        Values to replace with (i.e. "values")

    Returns
    -------
    np.array 
        Remapped array

    """
    assert len(k) == len(v), "k and v should have same length"
    mapping_ar = np.arange(arr.max()+1)
    mapping_ar[k] = v
    return mapping_ar[arr]


def downsample(arr, factor, axis):
    """Downsample a (2D) numpy array

    Parameters
    ----------
    arr : np.array
        Array to downsample
    factor : int 
        Factor to downsample by
    axis : int (0 or 1)
        Axis to downsample along

    Returns
    -------
    np.array 
        Downsampled array

    """
    assert axis == 0 or axis == 1, "axis should be 0 or 1"
    if axis == 0:
        return arr[::factor,:]
    elif axis == 1:
        return arr[:,::factor]





