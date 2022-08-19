"""ReachPlanner class

This module defines the ReachPlanner class.

"""

import numpy as np
import time 

import planeslam.planning.params as params
from planeslam.planning.LPM import LPM
import planeslam.planning.utils as utils


class ReachPlanner:
    """RTD Planner class

    Simple reachability-based trajectory planner.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, lpm_file, p_0, map, r_body):
        """Constructor
        
        Parameters
        ----------
        lpm_file : str
            File name for linear planning model 
        p_0 : np.array (N_DIM x 1)
            Initial position
        map : Scan
            Current map of environment
        r_body : Box3D
            Robot body (centered at 0)

        """
        # Initialize LPM object
        self.LPM = LPM(lpm_file)
        self.N_T_PLAN = len(self.LPM.time)  # planned trajectory length
        self.DT = self.LPM.t_sample  # trajectory discretization time interval

        # Initial conditions [m],[m/s],[m/s^2]
        self.p_0 = p_0
        self.v_0 = np.zeros((params.N_DIM,1))
        self.a_0 = np.zeros((params.N_DIM,1))

        # Current planned trajectory
        self.trajectory = utils.Trajectory(self.LPM.time, params.N_DIM)
        self.trajectory.P = np.tile(self.p_0, (1,self.N_T_PLAN))  # Initialize stationary

        # Goal position [m]
        self.p_goal = np.zeros((params.N_DIM,1))

        # Map (current merged scan)
        self.map = map

        # Robot body (represented as zonotope)
        self.r_body = r_body


    def check_plane_collision(self, positions, plane):
        """Check a sequence of positions against a single plane for collision.

        Parameters
        ----------
        positions : np.array (N_DIM x TRAJ_LEN)
            Sequence of positions (trajectory).
        plane : BoundedPlane
            Plane to check collision against.

        Returns
        -------
        bool
            True if there is a collision, False if not.

        """
        # TODO: vectorize this (both in speed and memory)
        # for i in range(self.N_T_PLAN):
        #     pos = positions[:,i]

        #     #start_time = time.time()
        #     #check = utils.collision_check(self.r_body + pos[:,None], plane.to_zonotope())
        #     #print(" check box plane time: ", time.time() - start_time)

        #     start_time = time.time()
        #     check = utils.check_box_plane_intersect(self.r_body.translate(pos), plane)
        #     #print(" check box plane time: ", time.time() - start_time)
        #     if check:
        #         return True
        # return False

        # TODO: try check line segments of trajectory with plane (don't consider robot volume)
        for i in range(self.N_T_PLAN-1):
            line_seg = positions[:,i:i+2].T

            check = plane.check_line_intersect(line_seg)
            #print(" check box plane time: ", time.time() - start_time)
            if check:
                return True
        return False


    def check_map_collisions(self, positions):
        """Check a sequence of positions against the current map for collision.

        Parameters
        ----------
        positions : np.array (N_DIM x TRAJ_LEN)
            Trajectory positions to check against obstacles.

        Returns
        -------
        bool
            True if there is a collision, False if not.

        """
        for plane in self.map.planes:
            # Only check planes within certain distance of initial position
            if plane.dist_to_point(self.p_0.flatten()) < params.COLLISION_CHECK_RADIUS:
                if self.check_plane_collision(positions, plane):
                    return True
        return False


    def traj_opt(self, t_start_plan):
        """Trajectory Optimization

        Attempt to find a collision-free plan (v_peak) which brings the agent 
        closest to its goal.

        Parameters
        ----------
        t_start_plan : float
            Time at which planning started for this iteration

        Returns
        -------
        np.array or None
            Optimal v_peak or None if failed to find one
        
        """

        # Generate potential v_peak samples
        V_peak = utils.rand_in_bounds(params.V_BOUNDS, params.N_PLAN_MAX)
        # Eliminate samples that exceed the max velocity and max delta from initial velocity
        V_peak = utils.prune_vel_samples(V_peak, self.v_0, params.V_MAX_NORM, params.DELTA_V_PEAK_MAX)

        # Calculate the endpoints for the sample v_peaks
        P_endpoints = self.LPM.compute_endpoints(self.v_0, self.a_0, V_peak) + self.p_0
        
        # Sort V_peaks by distance to goal
        dist_to_goal = np.linalg.norm(P_endpoints - self.p_goal, axis=0)
        V_sort_idxs = np.argsort(dist_to_goal)
        V_peak = V_peak[:,V_sort_idxs]

        # Iterate through V_peaks until we find a feasible one
        for i in range(V_peak.shape[1]):
        
            # Get candidate trajectory positions for current v_peak
            v_peak = V_peak[:,i][:,None]
            k = np.hstack((self.v_0, self.a_0, v_peak))
            cand_traj = self.LPM.compute_positions(k) + self.p_0

            # Check against obstacles
            start_time = time.time()
            check_collision = self.check_map_collisions(cand_traj)
            #print("check map collision time: ", time.time() - start_time)

            if not check_collision:
                #print("v_peak: ", v_peak, " idx = ", i)
                return v_peak

            if (time.time() - t_start_plan > params.T_PLAN):
                print("Ran out of time for planning, idx = ", i)
                break

        # No v_peaks are feasible (or we ran out of time)
        return None


    def replan(self, initial_conditions):
        """Replan

        Periodically called to perform trajectory optimization.

        Parameters
        ----------
        initial_conditions : Tuple


        Returns
        -------
        
        """
        t_start_plan = time.time()

        # Update initial conditions
        self.p_0, self.v_0, self.a_0 = initial_conditions

        # Find a new v_peak
        v_peak = self.traj_opt(t_start_plan)

        if v_peak is None:
            # Failed to find new plan
            return None
        else:
            # Generate new trajectory
            k = np.hstack((self.v_0, self.a_0, v_peak))
            P,V,A = self.LPM.compute_trajectory(k)
            P = P + self.p_0  # translate to p_0

            return P,V,A
