"""Utilities for trajectory planning

"""

import numpy as np
import time

from planeslam.planning.zonotope import is_empty_con_zonotope


class Trajectory:
    """Trajectory class

    Planned trajectory
    
    Attributes
    ----------
    length : int
        Length of trajectory (i.e. number of timesteps)
    N_dim : int
        State dimension of the trajectory (i.e. 2D or 3D)
    t : np.array (1 x N)
        Time array
    p : np.array (N_dim x N)
        Positions
    v : np.array (N_dim x N)
        Velocities
    a : np.array (N_dim x N)
        Accelerations

    """

    def __init__(self, T, N_dim):
        """Initialize a trajectory with 0 position, velocity, and acceleration"""
        self.T = T
        self.length = len(T)
        self.N_dim = N_dim
        self.P = np.zeros((N_dim, self.length))
        self.V = np.zeros((N_dim, self.length))
        self.A = np.zeros((N_dim, self.length))


def collision_check(Z1, Z2):
    """Check if two zonotopes collide (i.e. their intersection is nonempty)

    Method 1: find intersection as constrained zonotope, then check if that is empty
    Method 2: compute minkowski sum of Z1 and Z2 shifted to 0, and check if center of Z2 is contained inside

    Returns
    -------
    bool
        True if zonotopes intersect, False otherwise.
    
    """
    conZ_A = np.hstack((Z1.G, -Z2.G))
    conZ_b = Z2.c - Z1.c

    return not is_empty_con_zonotope(conZ_A, conZ_b)


def check_box_plane_intersect(box, plane):
    """Check if box intersects plane
    
    Parameters
    ----------
    box : Box3D
        Box to check intersection
    plane : BoundedPlane
        Plane to check intersection

    Returns
    -------
    bool
        True if intersecting, False otherwise
    
    """
    for e in box.edges():
        if plane.check_line_intersect(e):
            return True
    return False


def rand_in_bounds(bounds, n):
    """Generate random samples within specified bounds

    Parameters
    ----------
    bounds : list
        List of min and max values for each dimension.
    n : int
        Number of points to generate.

    Returns
    -------
    np.array 
        Random samples

    """
    x_pts = np.random.uniform(bounds[0], bounds[1], n)
    y_pts = np.random.uniform(bounds[2], bounds[3], n)
    # 2D 
    if len(bounds) == 4:
        return np.vstack((x_pts, y_pts))
    # 3D
    elif len(bounds) == 6:
        z_pts = np.random.uniform(bounds[4], bounds[5], n)
        return np.vstack((x_pts, y_pts, z_pts))
    else:
        raise ValueError('Please pass in bounds as either [xmin xmax ymin ymax] '
                            'or [xmin xmax ymin ymax zmin zmax] ')


def prune_vel_samples(V, v_0, max_norm, max_delta):
    """Prune Velocity Samples
    
    """
    V_mag = np.linalg.norm(V, axis=0)
    delta_V = np.linalg.norm(V - v_0, axis=0)
    keep_idx = np.logical_and(V_mag < max_norm, delta_V < max_delta)
    return V[:,keep_idx]


def dlqr_calculate(G, H, Q, R):
    """
    Discrete-time Linear Quadratic Regulator calculation.
    State-feedback control  u[k] = -K*x[k]
    Implementation from  https://github.com/python-control/python-control/issues/359#issuecomment-759423706
    How to apply the function:    
        K = dlqr_calculate(G,H,Q,R)
        K, P, E = dlqr_calculate(G,H,Q,R, return_solution_eigs=True)
    Inputs:
      G, H, Q, R  -> all numpy arrays  (simple float number not allowed)
      returnPE: define as True to return Ricatti solution and final eigenvalues
    Returns:
      K: state feedback gain
      P: Ricatti equation solution
      E: eigenvalues of (G-HK)  (closed loop z-domain poles)
      
    """
    from scipy.linalg import solve_discrete_are, inv
    P = solve_discrete_are(G, H, Q, R)  #Solução Ricatti
    K = inv(H.T@P@H + R)@H.T@P@G    #K = (B^T P B + R)^-1 B^T P A 

    return K