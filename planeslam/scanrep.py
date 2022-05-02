"""ScanRep class and utilities

This module defines the ScanRep class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt

import planeslam.general as general
import planeslam.mesh as mesh


class ScanRep:
    """Scan Representation class.

    This class represents a processed 3D LiDAR point cloud scan, in which the scan is stored
    as a set of vertices and faces.

    Attributes
    ----------
    vertices : np.array (n_verts x 3)
        Ordered array of vertices in scan
    faces : list of lists
        Sets of 4 vertex indices which form a face
    
    Methods
    -------
    plot()

    """
    vertices = []
    faces = []

    def __init__(self, vertices, faces, normals=None):
        """Construct scan object
        
        Parameters
        ----------
        center : np.array (3 x 1)
            Point at which scan is centered at (i.e. LiDAR pose position)
        orientation : 
            Orientation at which scan was taken at (i.e. LiDAR pose orientation)
        vertices : np.array (n_verts x 3)
            Ordered array of vertices in scan
        faces : list of lists
            Sets of 4 vertex indices which form a face, ordered CCW with respect to scan center
        normals : np.array (n_faces x 3)
            Set of normal vectors for each face

        """
        self.vertices = vertices
        self.faces = faces

        if normals is not None:
            self.normals = normals
        else:
            pass
            print("Computing normals not yet implemented")
            # Compute normals, if not given
            # Assumes vertex indicies for each face are given in CCW order (i.e. normal pointing "in" towards LiDAR)
            #self.normals 


    def transform(self, R, t):
        """Transform scan by rotation R and translation t

        Parameters
        ----------
        R : np.array (3 x 3)
            Rotation matrix
        t : np.array (1 x 3)
            Translation vector
        
        """
        self.vertices = (R @ self.vertices.T).T + t
    

    def plot(self, ax=None, color='b'):
        """Plot

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on, if not provided, will generate new set of axes
        color : optional
            Color to plot, default blue
        
        """
        if ax == None:
            fig, ax = plt.subplots()

        for f in self.faces:
            p = self.vertices[f,:]
            ax.plot(np.hstack((p[:,0],p[0,0])), np.hstack((p[:,1],p[0,1])), np.hstack((p[:,2],p[0,2])), color=color)


def pc_to_scan(P):
    """Point cloud to scan

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Unorganized point cloud

    Returns
    -------
    ScanRep
        Scan representing input point cloud
    
    """
    # Downsample
    P = general.downsample(P, factor=5, axis=0)

    # Generate mesh
    m = mesh.lidar_mesh(P)
    # Prune the mesh for long edges
    m = mesh.prune_mesh(P, m, 10)
    # Cluster the mesh with graph search
    clusters, avg_normals = mesh.cluster_mesh_graph_search(P, m)

    # Form scan topology
    vertices, faces, normals = mesh.scan_from_clusters(P, m, clusters, avg_normals)
    return ScanRep(vertices, faces, normals)