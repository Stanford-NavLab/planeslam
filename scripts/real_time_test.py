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

    # Initial position
    x_init = np.array([0, 0, 0])

    # Setup visualizations
    scan_vis = o3d.visualization.Visualizer()
    scan_vis.create_window()
    geoms = merged.o3d_geometries()
    for g in geoms:
        scan_vis.add_geometry(g)
    scan_vis.poll_events()
    scan_vis.update_renderer()

    traj_vis = o3d.visualization.Visualizer()
    traj_vis.create_window()
    traj_points = o3d.geometry.PointCloud()
    traj_lines = o3d.geometry.LineSet()
    traj_vis.add_geometry(traj_points)
    traj_vis.add_geometry(traj_lines)
    traj_vis.poll_events()
    traj_vis.update_renderer()

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
        t_abs = T_abs[:3,3].flatten()
        scans[i+1].transform(R_abs, t_abs)
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
        scan_vis.clear_geometries()
        for g in geoms:
            scan_vis.add_geometry(g)
        scan_vis.poll_events()
        scan_vis.update_renderer()

        traj_points.points = o3d.utility.Vector3dVector(np.vstack((np.asarray(traj_points.points), t_abs)))
        traj_lines.points = o3d.utility.Vector3dVector(np.vstack((np.asarray(traj_lines.points), t_abs)))
        traj_lines.lines = o3d.utility.Vector2iVector(np.vstack((np.asarray(traj_lines.lines), np.array([i,i+1]))))
        traj_vis.update_geometry(traj_points)
        traj_vis.update_geometry(traj_lines)
        traj_vis.poll_events()
        traj_vis.update_renderer()

        time.sleep(0.1)