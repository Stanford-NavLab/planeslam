"""Running full pipeline in real-time 

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import planeslam.io as io
from planeslam.scan import pc_to_scan
from planeslam.registration import decoupled_GN_register
from planeslam.clustering import mesh_cluster_pts

os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":

    # Read in point cloud data
    print("Reading in AirSim data...")
    binpath = os.path.join(os.getcwd(),'..', 'data', 'airsim', 'blocks_20_samples_1', 'lidar', 'Drone0')
    PCs = io.read_lidar_bin(binpath)
    # Read in ground-truth poses (in drone local frame)
    posepath = os.path.join(os.getcwd(),'..', 'data', 'airsim', 'blocks_20_samples_1', 'poses', 'Drone0')
    drone_positions, drone_orientations = io.read_poses(posepath)

    # Extract scans and planesets
    num_scans = len(PCs)
    scans = num_scans * [None]

    scans[0] = pc_to_scan(PCs[0])
    merged = scans[0]
    
    print("Beginning...")
    for i in range(len(scans)-1):
        print("i = ", i)
        # Extraction
        start_time = time.time()
        scans[i+1] = pc_to_scan(PCs[i+1])
        print("  extraction time: ", time.time() - start_time)

        # Registration
        start_time = time.time()
        R_hat, t_hat = decoupled_GN_register(scans[i], scans[i+1])
        print("  registration time: ", time.time() - start_time)
        
        # Merging
        start_time = time.time()
        merged = merged.merge(scans[i+1])
        
        print("  merge time: ", time.time() - start_time)