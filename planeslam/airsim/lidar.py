"""Functions for interfacing with AirSim lidar

"""

import airsim
import numpy as np


def get_lidar_PC(client):
    """Get Lidar Point cloud
    
    Parameters
    ----------
    client : airsim.MultirotorClient
        AirSim client object

    Returns
    -------
    P : np.array
        Point cloud

    """
    lidar_data = client.getLidarData()
    # time_stamp = lidar_data.time_stamp
    # lidar_pose = lidar_data.pose

    P = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
    P = np.reshape(P, (int(P.shape[0]/3), 3))
    
    return P#, lidar_pose, time_stamp



def reset(client):
    """Reset simulation state

    """
    print("Disarming...")
    client.armDisarm(False)
    print("Resetting")
    client.reset()
    client.enableApiControl(False)
