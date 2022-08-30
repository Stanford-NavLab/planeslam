"""Iterative visualization for clustering, plane extraction, and merging

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from planeslam.geometry.util import quat_to_R
from planeslam.general import plot_3D_setup, color_legend, NED_to_ENU
import planeslam.io as io
from planeslam.scan import pc_to_scan
from planeslam.registration import get_correspondences
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
    meshes = num_scans * [None]
    clusters = num_scans * [None]

    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, {'type': 'surface'}]])
    fig.update_layout(scene=dict(aspectmode='data'))
    
    f = go.FigureWidget(fig)

    f.show()

    #merged = scans[0]
    start_time = time.time()
    scans[0] = pc_to_scan(PCs[0])
    print("extraction time: ", time.time() - start_time)
    
    print("Beginning visualization...")
    for i in range(len(scans)-1):
        try:
            input("Waiting for keypress")
        except KeyboardInterrupt: 
            print("Exiting...")
            sys.exit(0)

        print("Extracting...")
        start_time = time.time()
        scans[i+1] = pc_to_scan(PCs[i+1])
        print("extraction time: ", time.time() - start_time)

        # Clear figure
        f.data = []

        # Plot new scans
        for t in scans[i].plot_trace():
            f.add_trace(t, row=1, col=1)

        for t in scans[i+1].plot_trace():
            f.add_trace(t, row=2, col=1)

        f.show()
        
        correspondences = get_correspondences(scans[i+1], scans[i])
        print("Correspondences: ", correspondences)