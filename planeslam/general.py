"""General utilities

"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


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
    P[:,[0,1]] = P[:,[1,0]]  # Swap x and y
    P[:,2] = -P[:,2]  # Negate z
    return P


def remap(arr, k, v):
    """Remap the elements of an array using the map k->v

    https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python

    Elements of arr which do not appear in k will be unchanged. Works for 
    k and v non-negative integers.

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
    # map = dict(zip(k,v))
    # return np.vectorize(map.get)(arr)


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


def adaptive_downsample(P, factor=5):
    """Downsample a point cloud based on distances
    
    Points farther away will get sampled with higher probability

    Parameters
    ----------
    P : np.array (N x 3)
        Point cloud to downsample
    factor : int 
        Factor to downsample by

    Returns
    -------
    np.array (N/factor x 3)
        Downsampled point cloud
    
    """
    np.random.seed(0)
    dists = np.linalg.norm(P, axis=1)
    keep_idxs = np.random.choice(np.arange(len(dists)), size=int(len(dists)/factor), replace=False, p=dists/np.sum(dists))
    return P[keep_idxs]


def normalize(v):
    """Normalize numpy vector

    Parameters
    ----------
    v : np.array 
        Vector to normalize

    Returns
    -------
    np.array 
        Normalized vector

    """
    return v / np.linalg.norm(v)


def plot_3D_setup(P=None, figsize=(15,10)):
    """Setup matplotlib 3D plotting

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Points to use for axis scaling
    
    Returns
    -------
    ax : axes
        Axes for plotting
    
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    if P is not None:
        ax.set_box_aspect((np.ptp(P[:,0]), np.ptp(P[:,1]), np.ptp(P[:,2])))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return ax


def color_legend(ax, num_colors):
    """Display legend for matplotlib default colors
    
    Parameters
    ----------
    ax : axes
        Axes to display on
    num_colors : int
        Number of colors in legend

    """
    for i in range(num_colors):
        ax.plot(0, 0, 0, color='C'+str(i), label=str(i))
    ax.legend()


def pc_plot_trace(P, color=None):
    """Generate plotly plot trace for point cloud

    Parameters
    ----------
    P : np.array (N x 3)
        Point cloud

    Returns
    -------
    go.Scatter3d
        Scatter plot trace

    """
    return go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2], 
        mode='markers', marker=dict(size=2, color=color))


def pose_plot_trace(R, t):
    """Generate plotly plot trace for a 3D pose (position and orientation)

    Parameters
    ----------
    R : np.array (3 x 3)
        Rotation matrix representing orientation
    t : np.array (3)
        Translation vector representing position
    
    Returns
    -------
    list
        List containing traces for plotting 3D pose

    """
    point = go.Scatter3d(x=[t[0]], y=[t[1]], z=[t[2]], 
                mode='markers', marker=dict(size=5))
    xs = []; ys = []; zs = []
    for i in range(3):
        xs += [t[0], t[0] + R[0,i], None]
        ys += [t[1], t[1] + R[1,i], None]
        zs += [t[2], t[2] + R[2,i], None]
    lines = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", showlegend=False)
    return [point, lines]


def trajectory_plot_trace(Rs, ts, color="red"):
    """Generate plotly plot trace for a 3D trajectory of poses

    Parameters
    ----------
    Rs : np.array (3 x 3 x N)
        Sequence of orientations
    ts : np.array (N x 3)
        Sequence of positions
    
    Returns
    -------
    list
        List containing traces for plotting 3D trajectory
    
    """
    points = go.Scatter3d(x=[ts[:,0]], y=[ts[:,1]], z=[ts[:,2]], showlegend=False)#, mode='markers', marker=dict(size=5))
    xs = []; ys = []; zs = []
    for i in range(len(ts)):
        for j in range(3):
            xs += [ts[i,0], ts[i,0] + Rs[0,j,i], None]
            ys += [ts[i,1], ts[i,1] + Rs[1,j,i], None]
            zs += [ts[i,2], ts[i,2] + Rs[2,j,i], None]
    lines = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color), showlegend=False)
    return [points, lines]
    

def plot_normals(P,normals,scale=10.):
    """Plot point cloud using plotly

    Returns
    -------
    fig : plotly go.Figure
        Figure handle

    """
    n = P.shape[0]
    points = go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2], 
        mode='markers', marker=dict(size=2))
    data = [points]
    xs = [None for i in range(3*n)]
    ys = [None for i in range(3*n)]
    zs = [None for i in range(3*n)]
    for i in range(n):
        xs[3*i] = P[i,0]
        xs[3*i+1] = P[i,0]+scale*normals[i,0]
        ys[3*i] = P[i,1]
        ys[3*i+1] = P[i,1]+scale*normals[i,1]
        zs[3*i] = P[i,2]
        zs[3*i+1] = P[i,2]+scale*normals[i,2]
    data.append(go.Scatter3d(x=xs,y=ys,z=zs,mode="lines"))

    fig = go.Figure(data=data)
    fig.update_layout(width=1000, height=600, scene=dict(
                    aspectmode='data'),showlegend=False)
    return fig

