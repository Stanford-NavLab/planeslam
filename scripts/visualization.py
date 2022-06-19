"""Iterative visualization for clustering, plane extraction, and merging

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from planeslam.geometry.util import quat_to_rot_mat
from planeslam.general import plot_3D_setup, color_legend, NED_to_ENU
import planeslam.io as io
from planeslam.extraction import pc_to_planes
from planeslam.scan import pc_to_scan
from planeslam.registration import get_correspondences


if __name__ == "__main__":

    # Read in point cloud data
    print("Reading in AirSim data...")
    binpath = os.path.join(os.getcwd(),'..', 'data', 'airsim', 'building_99_60_samples_kitchen_corridor', 'lidar', 'Drone0')
    PCs = io.read_lidar_bin(binpath)
    # Read in ground-truth poses (in drone local frame)
    posepath = os.path.join(os.getcwd(),'..', 'data', 'airsim', 'building_99_60_samples_kitchen_corridor', 'poses', 'Drone0')
    drone_positions, drone_orientations = io.read_poses(posepath)

    # Extract scans and planesets
    num_scans = len(PCs)
    scans = num_scans * [None]
    planesets = num_scans * [None]

    print("Extracting scans...")
    for i in range(num_scans):
        PCs[i] = NED_to_ENU(PCs[i])
        scans[i] = pc_to_scan(PCs[i])
        print(i, "...")
    
    plt.ion()
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    
    print("Beginning visualization...")
    for i in range(len(scans)-1):
        try:
            input("Waiting for keypress")
        except KeyboardInterrupt: 
            print("Exiting...")
            sys.exit(0)

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        # Plot point clouds
        ax1.scatter(PCs[i][:,0], PCs[i][:,1], PCs[i][:,2], marker='.', s=1)  
        ax2.scatter(PCs[i+1][:,0], PCs[i+1][:,1], PCs[i+1][:,2], marker='.', s=1)  

        ax1.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        ax2.set_box_aspect((np.ptp(PCs[i+1][:,0]), np.ptp(PCs[i+1][:,1]), np.ptp(PCs[i+1][:,2])))
        ax1.set_title("Scan "+str(i))
        ax2.set_title("Scan "+str(i+1))

        # Plot scans
        scans[i].plot(ax3)
        scans[i+1].plot(ax4)
        color_legend(ax3, len(scans[i].planes))
        color_legend(ax4, len(scans[i+1].planes))
        ax3.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        ax4.set_box_aspect((np.ptp(PCs[i+1][:,0]), np.ptp(PCs[i+1][:,1]), np.ptp(PCs[i+1][:,2])))
        
        correspondences = get_correspondences(scans[i+1], scans[i])
        print("Correspondences: ", correspondences)

        fig.canvas.draw()
        fig.canvas.flush_events()