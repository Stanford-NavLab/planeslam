"""Utilities for I/O

"""

import numpy as np
import struct 
import os


def read_lidar_bin(binpath):
    """Read bin files containing LiDAR point cloud data 

    Parameters
    ----------
    binpath : str
        Path to folder containing bin files

    Returns
    -------
    PC_data : list of np.array (n_pts x 3)
        List of point clouds for each frame

    """
    PC_data = []
    size_float = 4
    for i in range(1,len(os.listdir(binpath))+1):
        list_pcd = []
        bf = binpath+"/{i}.bin".format(i=str(i).zfill(6))

        try: 
            with open(bf, "rb") as f: 
                byte = f.read(size_float*4)
                while byte:
                    x, y ,z, intensity = struct.unpack("ffff", byte)
                    list_pcd.append([x, y, z])
                    byte = f.read(size_float * 4)
            np_pcd = np.asarray(list_pcd)
        except FileNotFoundError:
            print(f"file {i} wasn't found")
        
        PC_data.append(np_pcd)
    
    return PC_data


def read_poses(path):
    """Read txt files containing pose information

    Pose is stored as x,y,z position and quaternion orientation

    Parameters
    ----------
    path : str
        Path to folder containing txt files

    Returns
    -------
    positions : np.array (n_frames x 3)
        x,y,z position for each frame
    orientations : np.array (n_frames x 4)
        quaternion orientation for each frame

    """
    positions = []
    orientations = []
    for i in range(1,len(os.listdir(path))+1):
        pf = path+"/{i}.txt".format(i=str(i).zfill(6))

        with open(pf) as f:
            lines = f.read().splitlines()
            pos = [float(x) for x in lines[0].split(',')]
            quat = [float(x) for x in lines[1].split(',')]
            positions.append(pos)
            orientations.append(quat)
    
    positions = np.asarray(positions)
    orientations = np.asarray(orientations)
    return positions, orientations