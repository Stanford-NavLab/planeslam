"""Iterative visualization for clustering, plane extraction, and merging

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from planeslam.geometry.util import quat_to_rot_mat
from planeslam.general import plot_3D_setup, color_legend, NED_to_ENU
import planeslam.io as io
from planeslam.scan import pc_extraction
from planeslam.registration import get_correspondences
from planeslam.clustering import mesh_cluster_pts

os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":

    # Read in point cloud data
    print("Reading in AirSim data...")
    binpath = os.path.join(os.getcwd(),'..', 'data', 'airsim', 'blocks_60_samples_loop_closure', 'lidar', 'Drone0')
    PCs = io.read_lidar_bin(binpath)
    # Read in ground-truth poses (in drone local frame)
    posepath = os.path.join(os.getcwd(),'..', 'data', 'airsim', 'blocks_60_samples_loop_closure', 'poses', 'Drone0')
    drone_positions, drone_orientations = io.read_poses(posepath)

    # Extract scans and planesets
    num_scans = len(PCs)
    scans = num_scans * [None]
    meshes = num_scans * [None]
    clusters = num_scans * [None]
    
    plt.ion()
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.set_xlabel("X"); ax1.set_ylabel("Y")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y")

    start_time = time.time()
    meshes[0], clusters[0], scans[0] = pc_extraction(PCs[0])
    print("extraction time: ", time.time() - start_time)
    scans[0].transform(quat_to_rot_mat(drone_orientations[0,:]), drone_positions[0,:])
    merged = scans[0]
    
    print("Beginning visualization...")
    for i in range(len(scans)-1):
        try:
            input("Waiting for keypress")
        except KeyboardInterrupt: 
            print("Exiting...")
            sys.exit(0)

        print("Extracting...")
        start_time = time.time()
        meshes[i+1], clusters[i+1], scans[i+1] = pc_extraction(PCs[i+1])
        print("extraction time: ", time.time() - start_time)
        scans[i+1].transform(quat_to_rot_mat(drone_orientations[i+1,:]), drone_positions[i+1,:])

        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Plot scans
        merged.plot(ax1)
        scans[i+1].plot(ax2)
        merged = merged.merge(scans[i+1])
        merged.plot(ax3)
        ax1.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        ax2.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        ax3.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        
        correspondences = get_correspondences(scans[i+1], scans[i])
        print("Correspondences: ", correspondences)

        fig.canvas.draw()
        fig.canvas.flush_events()