# Linear Planning Model Class

import numpy as np
from scipy.io import loadmat

class LPM:
    """Linear Planning Model class.

    This class represents the Linear Planning Model, which converts trajectory parameters 
    to trajectories. It contains attributes for the trajectory parameterization.

    Attributes
    ----------
    t_peak : float
        Time in trajectory corresponding to peak velocity
    t_total : float
        Total time duration of trajectory
    t_sample : float
        Sample time of trajectory (discretization)
    time : np.array
        Time vector for trajectory
    P_mat : np.array (3 x N)
        Matrix for computing positions
    V_mat : np.array (3 x N)
        Matrix for computing velocities
    A_mat : np.array (3 x N)
        Matrix for computing accelerations
    
    Methods
    -------
    compute_trajectory(k) 
        Compute nominal trajectory from a given trajectory parameter
    solve_trajectory(v_0, a_0, p_goal)

    """
    def __init__(self, mat_file):
        """Construct LPM object from .mat file.

        Parameters
        ----------
            mat_file : .mat
                .mat file containing all the LPM parameters

        """
        # Load the .mat file
        lpm = loadmat(mat_file)
        lpm = lpm['LPM']

        # Extract variables, convert arrays to numpy arrays
        self.t_peak = lpm['t_peak'][0,0][0][0]
        self.t_total = lpm['t_total'][0,0][0][0]
        self.t_sample = lpm['t_sample'][0,0][0][0]
        self.time = np.array(lpm['time'][0,0])[0]
        self.P_mat = np.array(lpm['position'][0,0])
        self.V_mat = np.array(lpm['velocity'][0,0])
        self.A_mat = np.array(lpm['acceleration'][0,0])


    def compute_trajectory(self, k):
        """Compute nominal trajectory from a given trajectory parameter.

        Parameters
        ----------
        k : np.array
            trajectory parameter k = (v_0, a_0, v_peak), n x 3 where n is workspace dimension
        
        Returns
        -------
        Tuple 
            Tuple of the form (p,v,a) where each of p,v,a are n x N where n is the workspace dimension.
        
        """
        p = k @ self.P_mat
        v = k @ self.V_mat
        a = k @ self.A_mat
        return p,v,a

    
    def compute_positions(self, k):
        """Compute positions from a given trajectory parameter.

        Parameters
        ----------
        k : np.array
            trajectory parameter k = (v_0, a_0, v_peak), n x 3 where n is workspace dimension
        
        Returns
        -------
        np.array 
            Positions
        
        """
        return k @ self.P_mat

    
    def compute_endpoints(self, v_0, a_0, V_peak):
        """Compute trajectory endpoints given initial conditions and collection of V_peaks
        
        """
        LPM_p_final = self.P_mat[:,-1]
        # Final position contribution from v_0 and a_0
        p_from_v_0_and_a_0 = (np.hstack((v_0, a_0)) @ LPM_p_final[:2])[:, None]
        # Add in contribution from v_peak
        P_endpoints = p_from_v_0_and_a_0 + LPM_p_final[2] * V_peak 
        return P_endpoints


    def solve_trajectory(self, v_0, a_0, p_goal):
        """Solve for the peak velocity which reaches a desired goal position.

        Note that this solution does not account for max velocity constraints

        Parameters
        ----------
        v_0 : np.array
            Initial velocity (1 x n row vector)
        a_0 : np.array
            Initial acceleration (1 x n row vector)
        p_goal : np.array
            Goal position (1 x n row vector)

        Returns
        -------
        np.array 
            Peak velocity (1 x n row vector).
            
        """
        # Change to column vectors
        v_0 = np.reshape(v_0, (3,1))
        a_0 = np.reshape(a_0, (3,1))
        # Position component due to v_0 and a_0
        p_from_ic = np.dot(np.hstack((v_0, a_0)), self.P_mat[0:2,-1])
        # Solve for v_peak
        v_peak = (p_goal - p_from_ic) / self.P_mat[2,-1]
        return v_peak

