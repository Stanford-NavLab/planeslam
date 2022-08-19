"""Reachability functions

"""

import numpy as np
import time

import planeslam.planning.params as params
from planeslam.planning.zonotope import Zonotope


def compute_PRS(LPM, v_0, a_0, V_pk):
    """Compute Planning Reachable Set (PRS)
    
    PRS desribes the reachable positions of planned trajectories over a 
    space of chosen trajectory parameters (v_peak). These sets do not account  
    for any deviations from the planned trajectories to the actual trajectory
    executed by the robot (this is handled by the Error Reachable Set or ERS).
    They can however, model uncertainty in initial conditions v_0 and a_0 
    (e.g. provided by state estimator covariance).

    PRS is always centered at 0 and is of dimension 2*N_DIM. It has N_DIM 
    dimensions for position, and N_DIM dimensions for peak velocities, so that
    it can later be sliced in the peak velocity dimensions for trajectory planning.

    Parameters
    ----------
    LPM : LPM 
        Linear planning model object
    v_0 : np.array (N_DIM)
        Initial velocity
    a_0 : np.array (N_DIM)
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

    k_0 = np.vstack((v_0, a_0))
    p_0 = LPM.P_mat.T[:,:2] @ k_0  # trajectory due to initial conditions only

    PRS[0] = Zonotope(np.zeros((2*params.N_DIM,1)), np.zeros((2*params.N_DIM,1)))

    # For now, we only consider fixed v_0 and a_0
    # TODO: handle zonotope v_0 and a_0
    for i in range(1,N):
        pos = LPM.P_mat[2,i] * V_pk  # position zonotope
        PRS[i] = pos.augment(V_pk) + np.vstack((p_0[i][:,None], np.zeros((params.N_DIM,1))))

    return PRS


def generate_collision_constraints(FRS, obs):
    """Generate collision constraints

    Given an FRS and a set of obstacles, generate halfspace constraints
    in trajectory parameter space that constrain safe trajectory parameters.

    Parameters
    ----------
    FRS : list
        List of zonotopes representing forward reachable set
    obs : list
        List of zonotopes representing obstacles

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
    for i in range(1,len(FRS)):
        # For each obstacle
        for O in obs:
            # Get current obstacle
            
            # Extract center and generators of FRS
            