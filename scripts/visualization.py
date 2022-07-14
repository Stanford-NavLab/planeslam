"""Iterative visualization for clustering, plane extraction, and merging

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from planeslam.geometry.util import quat_to_rot_mat
from planeslam.general import plot_3D_setup, color_legend, NED_to_ENU
import planeslam.io as io
from planeslam.scan import pc_extraction
from planeslam.registration import get_correspondences
from planeslam.clustering import mesh_cluster_pts


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

    # print("Extracting scans...")
    # for i in range(num_scans):
    #     meshes[i], clusters[i], scans[i] = pc_extraction(PCs[i])
    #     print(i, "...")
        # R = quat_to_rot_mat(drone_orientations[i,:])
        # t = drone_positions[i,:]
        # scans[i].transform(R, t)
        # PCs[i] = (R @ PCs[i].T).T + t
    
    plt.ion()
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    ax5 = fig.add_subplot(2, 4, 5, projection='3d')
    ax6 = fig.add_subplot(2, 4, 6, projection='3d')
    ax7 = fig.add_subplot(2, 4, 7, projection='3d')
    ax8 = fig.add_subplot(2, 4, 8, projection='3d')

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax7.set_xlabel("X")
    ax7.set_ylabel("Y")
    ax8.set_xlabel("X")
    ax8.set_ylabel("Y")

    #merged = scans[0]

    meshes[0], clusters[0], scans[0] = pc_extraction(PCs[0])
    
    print("Beginning visualization...")
    for i in range(len(scans)-1):
        try:
            input("Waiting for keypress")
        except KeyboardInterrupt: 
            print("Exiting...")
            sys.exit(0)

        print("Extracting...")
        meshes[i+1], clusters[i+1], scans[i+1] = pc_extraction(PCs[i+1])

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        ax7.clear()
        ax8.clear()

        # Plot point clouds
        ax1.scatter(PCs[i][:,0], PCs[i][:,1], PCs[i][:,2], marker='.', s=1)  
        ax5.scatter(PCs[i+1][:,0], PCs[i+1][:,1], PCs[i+1][:,2], marker='.', s=1)  

        ax1.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        ax5.set_box_aspect((np.ptp(PCs[i+1][:,0]), np.ptp(PCs[i+1][:,1]), np.ptp(PCs[i+1][:,2])))
        ax1.set_title("Scan "+str(i))
        ax5.set_title("Scan "+str(i+1))

        # Plot meshes
        ax2.plot_trisurf(PCs[i][:,0], PCs[i][:,1], PCs[i][:,2], triangles=meshes[i].DT.simplices)
        ax6.plot_trisurf(PCs[i+1][:,0], PCs[i+1][:,1], PCs[i+1][:,2], triangles=meshes[i+1].DT.simplices)
        ax2.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        ax6.set_box_aspect((np.ptp(PCs[i+1][:,0]), np.ptp(PCs[i+1][:,1]), np.ptp(PCs[i+1][:,2])))

        # Plot clusterings
        for j, c in enumerate(clusters[i]):
            cluster_pts = mesh_cluster_pts(meshes[i], c)
            ax3.scatter3D(cluster_pts[:,0], cluster_pts[:,1], cluster_pts[:,2], color='C'+str(j), marker='.', s=1)
        for j, c in enumerate(clusters[i+1]):
            cluster_pts = mesh_cluster_pts(meshes[i+1], c)
            ax7.scatter3D(cluster_pts[:,0], cluster_pts[:,1], cluster_pts[:,2], color='C'+str(j), marker='.', s=1)
        ax3.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        ax7.set_box_aspect((np.ptp(PCs[i+1][:,0]), np.ptp(PCs[i+1][:,1]), np.ptp(PCs[i+1][:,2])))

        # Plot scans
        scans[i].plot(ax4)
        # merged = merged.merge(scans[i+1])
        # merged.plot(ax3)
        scans[i+1].plot(ax8)
        color_legend(ax4, len(scans[i].planes))
        color_legend(ax8, len(scans[i+1].planes))
        ax4.set_box_aspect((np.ptp(PCs[i][:,0]), np.ptp(PCs[i][:,1]), np.ptp(PCs[i][:,2])))
        ax8.set_box_aspect((np.ptp(PCs[i+1][:,0]), np.ptp(PCs[i+1][:,1]), np.ptp(PCs[i+1][:,2])))
        
        correspondences = get_correspondences(scans[i+1], scans[i])
        print("Correspondences: ", correspondences)

        fig.canvas.draw()
        fig.canvas.flush_events()