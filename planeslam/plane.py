"""BoundedPlane class and utilities

This module defines the BoundedPlane class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt

import planeslam.general as general
import planeslam.mesh as mesh


class BoundedPlane:
    """Bounded Plane class.

    This class represents a rectangularly bounded plane, represented by 4 coplanar 
    vertices which form a rectangle. 

    Attributes
    ----------
    vertices : np.array (4 x 3)
        Ordered array of vertices 
    
    Methods
    -------
    plot()

    """

    def __init__(self, vertices):
        """Construct plane
        
        Parameters
        ----------
        vertices : np.array (4 x 3)
            Ordered array of vertices 

        """
        self.vertices = vertices
        


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
        self.normals = (R @ self.normals.T).T
    

    def plot(self, ax=None, color='b', show_normals=False):
        """Plot

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on, if not provided, will generate new set of axes
        color : optional
            Color to plot, default blue
        show_normals : bool, optional
            Whether to plot normal vectors for each face
        
        """
        if ax == None:
            fig, ax = plt.subplots()

        for idx, f in enumerate(self.faces):
            p = self.vertices[f,:]
            ax.plot(np.hstack((p[:,0],p[0,0])), np.hstack((p[:,1],p[0,1])), np.hstack((p[:,2],p[0,2])), color=color)

            if show_normals:
                c = np.mean(p, axis=0)
                n = 10 * self.normals[idx]  # TODO: quiver scaling is currently arbitrary
                ax.quiver(c[0], c[1], c[2], n[0], n[1], n[2], color=color)


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