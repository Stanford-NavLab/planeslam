"""Running full pipeline in real-time 

"""

import numpy as np
import os
import sys
import time
import open3d as o3d

import planeslam.io as io
from planeslam.scan import pc_to_scan
from planeslam.registration import decoupled_GN_register
from planeslam.clustering import mesh_cluster_pts
from planeslam.general import NED_to_ENU

os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":

    # Read in point cloud data
    print("Reading in AirSim data...")
    binpath = os.path.join(os.getcwd(),'..', 'data', 'airsim', 'blocks_20_samples_1', 'lidar', 'Drone0')
    PCs = io.read_lidar_bin(binpath)
    # Read in ground-truth poses (in drone local frame)
    posepath = os.path.join(os.getcwd(),'..', 'data', 'airsim', 'blocks_20_samples_1', 'poses', 'Drone0')
    drone_positions, drone_orientations = io.read_poses(posepath)

    # Convert points to ENU
    num_scans = len(PCs)
    for i in range(num_scans):
        PCs[i] = NED_to_ENU(PCs[i])
    scans = num_scans * [None]

    scans[0] = pc_to_scan(PCs[0])
    merged = scans[0]

    # Setup visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geoms = merged.o3d_geometries()
    for g in geoms:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()

    T_abs = np.eye(4)
    
    input("Press any key to begin")
    
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
        T_hat = np.vstack((np.hstack((R_hat, t_hat)), np.hstack((np.zeros(3), 1))))
        T_abs = T_hat @ T_abs
        R_abs = T_abs[:3,:3]
        t_abs = T_abs[:3,3]
        scans[i+1].transform(R_abs, t_abs.flatten())
        print("  registration time: ", time.time() - start_time)
        
        # Merging
        start_time = time.time()
        merged = merged.merge(scans[i+1])
        merged.reduce_inside()
        merged.remove_small_planes()
        merged.fuse_edges()
        print("  merge time: ", time.time() - start_time)

        # Update visualization
        geoms = merged.o3d_geometries()
        # for g in geoms:
        #     vis.update_geometry(g)
        vis.clear_geometries()
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.1)