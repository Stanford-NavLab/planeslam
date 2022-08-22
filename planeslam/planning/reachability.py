"""Reachability functions

"""

import numpy as np

import planeslam.planning.params as params
from planeslam.planning.zonotope import Zonotope
from planeslam.general import remove_zero_columns


def compute_PRS(LPM, p_0, v_0, a_0):
    """Compute Planning Reachable Set (PRS)
    
    PRS desribes the reachable positions of planned trajectories over a 
    space of chosen trajectory parameters (v_peak). These sets do not account  
    for any deviations from the planned trajectories to the actual trajectory
    executed by the robot (this is handled by the Error Reachable Set or ERS).
    They can however, model uncertainty in initial conditions v_0 and a_0 
    (e.g. provided by state estimator covariance).

    PRS is of dimension 2*N_DIM. It has N_DIM dimensions for position, and N_DIM 
    dimensions for peak velocities, so that it can later be sliced in the peak 
    velocity dimensions for trajectory planning.

    NOTE: for now, only use position as state. Need to investigate whether velocity 
          as state is beneficial

    PRS dimensions:
      0 - x  
      1 - y  
      2 - z
      3 - v_pk_x
      4 - v_pk_y
      5 - v_pk_z


    Parameters
    ----------
    LPM : LPM 
        Linear planning model object
    p_0 : np.array (N_DIM x 1)
        Initial velocity
    v_0 : np.array (N_DIM x 1)
        Initial velocity
    a_0 : np.array (N_DIM x 1)
        Initial acceleration
    V_pk : Zonotope
        Zonotope of candidate peak velocities which parameterize possible planned trajectories

    Returns
    -------
    PRS : list
        List of Zonotope objects describing reachable positions from planned trajectories

    """
    # Initialize
    N = len(LPM.time)
    PRS = N * [None]

    k_0 = np.hstack((v_0, a_0)).T
    init_traj = LPM.P_mat.T[:,:2] @ k_0 + p_0.flatten()  # trajectory due to initial conditions only

    # Form trajectory parameter space zonotope
    # V_pk_c = np.vstack((k_0, np.zeros(params.N_DIM)))
    # V_pk_c = V_pk_c.reshape((-1,1), order='F')
    # V_pk_G = np.kron(np.eye(params.N_DIM), np.array([0,0,params.V_MAX])[:,None])
    #print(V_pk_c, V_pk_G)
    V_pk = Zonotope(np.zeros((params.N_DIM,1)), params.V_MAX * np.eye(params.N_DIM))

    PRS[0] = Zonotope(np.zeros((2*params.N_DIM,1)), np.zeros((2*params.N_DIM,1)))

    # For now, we only consider fixed v_0 and a_0
    # TODO: handle zonotope v_0 and a_0
    for i in range(1,N):
        pos = LPM.P_mat[2,i] * V_pk  # position zonotope
        PRS[i] = pos.augment(V_pk) + np.vstack((init_traj[i][:,None], np.zeros((params.N_DIM,1))))

    return PRS


def generate_collision_constraints(FRS, obs_map):
    """Generate collision constraints

    Given an FRS and a set of obstacles, generate halfspace constraints
    in trajectory parameter space that constrain safe trajectory parameters.

    Parameters
    ----------
    FRS : list
        List of zonotopes representing forward reachable set
    obs_map : list
        List of zonotopes representing obstacles
    k_dim : np.array (1D)
        Dimensions of trajectory parameter 
    obs_dim : np.array (1D)
        Dimensions of obstacle

    Returns
    -------
    A_con : list
        List of constraint matrices
    b_con : list 
        List of constraint vectors 

    """
    A_con = []
    b_con = []

    # Skip first time step since it is not sliceable
    for z in FRS[1:]:
        # Extract center and generators of FRS
        c = z.c[params.OBS_DIM]
        G = z.G

        # Find columns of G which are nonzero in k_dim ("k-sliceable")
        # - this forms a linear map from the parameter space to workspace
        k_col = list(set(np.nonzero(G[params.K_DIM,:])[1]))
        k_slc_G = G[params.OBS_DIM][:,k_col]

        # "non-k-sliceable" generators - have constant contribution regardless of chosen trajectory parameter
        k_no_slc_G = G[params.OBS_DIM]
        k_no_slc_G = np.delete(k_no_slc_G, k_col, axis=1)

        # For each obstacle
        for obs in obs_map:
            # Get current obstacle
            obs = obs.Z

            # Obstacle is "buffered" by non-k-sliceable part of FRS
            buff_obs_c = obs[:,0][:,None] - c
            buff_obs_G = np.hstack((obs[:,1:], k_no_slc_G))
            buff_obs_G = remove_zero_columns(buff_obs_G)
            buff_obs = Zonotope(buff_obs_c, buff_obs_G)

            A_obs, b_obs = buff_obs.halfspace()
            A_con.append(A_obs @ k_slc_G)  # map constraints to be over coefficients of k_slc_G generators
            b_con.append(b_obs)
    
    return A_con, b_con


def check_collision_constraints(A_con, b_con, v_peak):
    """Check a trajectory parameter against halfspace collision constraints.
    
    Parameters
    ----------
    A_con : list
        List of halfspace constraint matrices
    b_con : list
        List of halfspace constraint vectors
    v_peak : np.array (N_DIM x 1)
        Trajectory parameter

    Returns
    -------
    bool
        True if the trajectory is safe, False if it results in collision

    """
    c = np.inf

    # Get the coefficients of the parameter space zonotope for this parameter
    # Assumes parameter space zonotope is centered at 0, v_max generators
    lambdas = v_peak / params.V_MAX

    for (A, b) in zip(A_con, b_con):
        c_tmp = A @ lambdas - b  # A*lambda - b <= 0 means inside unsafe set
        c_tmp = max(c_tmp.flatten())  # Max of this <= 0 means inside unsafe set
        c = min(c, c_tmp)  # Find smallest max. If it's <= 0, then parameter is unsafe
    
    return c > 0
