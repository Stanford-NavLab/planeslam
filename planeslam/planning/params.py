import numpy as np

# Timing parameters
T_REPLAN = 10.0  # [s] amount of time between replans
T_PLAN = 10.0  # [s] amount of time allotted for planning itself 
              #     (allow buffer for actual tranmission of plan)

# System
N_DIM = 3  # workspace dimension (i.e. 2D or 3D)

# Max velocity constraints [m/s]
V_MAX = 5.0  # L1 velocity constraints
V_BOUNDS = np.tile(np.array([-V_MAX, V_MAX]), (1,N_DIM))[0]
V_MAX_NORM = 10.0  # L2 velocity constraints
DELTA_V_PEAK_MAX = 6.0  # Delta from initial velocity constraint

# Planning
R_GOAL_REACHED = 0.3  # [m] stop planning when within this dist of goal
COLLISION_CHECK_RADIUS = 10  # [m] radius for which to check planes in collision
N_PLAN_MAX = 10000  # Max number of plans to evaluate

# Reachability
K_DIM = np.array([3, 4, 5])  # Trajectory parameter dimensions (row idxs) in FRS
                              # TODO: generalize this
OBS_DIM = np.arange(N_DIM)  # Obstacle dimensions (row idxs) in FRS
                            # - obstacles exist in position space