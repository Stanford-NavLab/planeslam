"""LidarMesh class

This module defines the LidarMesh class and relevant utilities.

"""

import numpy as np
from scipy.spatial import Delaunay

import planeslam.general as general


class LidarMesh:
    """LidarMesh class.

    This class represents a mesh created from a LiDAR point cloud, and provides
    functions for operating on the mesh. 

    Attributes
    ----------
    P : np.array (n_pts x 3)
        Point cloud points
    DT : scipy.spatial.Delaunay
        Delaunay triangulation data structure 
    tri_nbr_dict : dictionary
        Dictionary of triangle neighbors
    
    
    Methods
    -------
    

    """

    def __init__(self, P):
        """Construct a mesh from an unorganized LiDAR point cloud using Delaunay triangulation

        Parameters
        ----------
        P : np.array (n_pts x 3)
            Point cloud points

        """
        self.P = P

        # Map points to 2D (with inverse spherical projection)
        # TODO: handle wrapping
        thetas = np.arctan2(P[:,1], P[:,0])
        Rxy = np.sqrt(P[:,0]**2 + P[:,1]**2)
        phis = np.arctan2(P[:,2], Rxy)

        # Generate Delaunay triangulation
        self.DT = Delaunay(np.stack((thetas,phis), axis=1))
        self.tri_nbr_dict = self.create_tri_nbr_dict()
    

    def prune(self, edge_len_lim):
        """Prune mesh by removing triangles with edge length exceeding specified length

        Parameters
        ----------
        edge_len_lim : float
            Maximum edge length to retain

        """
        # Compute side lengths
        T = self.P[self.DT.simplices]
        S1 = np.linalg.norm(T[:,0,:] - T[:,1,:], axis=1)  # Side 1 lengths
        S2 = np.linalg.norm(T[:,1,:] - T[:,2,:], axis=1)  # Side 2 lengths
        S3 = np.linalg.norm(T[:,2,:] - T[:,0,:], axis=1)  # Side 3 lengths
        keep_idx_mask = (S1 < edge_len_lim) & (S2 < edge_len_lim) & (S3 < edge_len_lim) 

        # Prune the simplices
        self.DT.simplices = self.DT.simplices[keep_idx_mask]

        # Update other fields of DT data stucture
        self.DT.equations = self.DT.equations[keep_idx_mask]
        
        # Remap indices for neighbors
        # Add 1 to all idxs to shift -1 to 0 (because remap operates on nonnegative values)
        self.DT.neighbors += 1
        full_idxs = np.arange(1,len(keep_idx_mask)+1)
        keep_idxs = full_idxs[keep_idx_mask]
        discard_idxs = full_idxs[~keep_idx_mask]
        
        if len(keep_idxs) < len(keep_idx_mask):
            # Remap discard idxs to 0
            self.DT.neighbors = general.remap(self.DT.neighbors, discard_idxs, np.zeros(len(discard_idxs)))
            # Remap keep idxs to start at 1
            self.DT.neighbors = general.remap(self.DT.neighbors, keep_idxs, np.arange(1,len(keep_idxs)+1))
        # Remove entries for deleted triangles
        self.DT.neighbors = self.DT.neighbors[keep_idx_mask]

        # Shift 0 back to -1
        self.DT.neighbors -= 1

        # Update tri_nbr_dict
        self.tri_nbr_dict = self.create_tri_nbr_dict()


    def vertex_neighbors(self, vertex):
        """Retrieve neighbors of a vertex in the mesh

        Parameters
        ----------
        vertex : int
            Vertex ID (index)

        Returns
        -------
        list
            List of neighbors

        """
        idx_ptrs, idxs = self.DT.vertex_neighbor_vertices
        return idxs[idx_ptrs[vertex]:idx_ptrs[vertex+1]]


    def create_tri_nbr_dict(self):
        """Create dictionary storing triangle neighbors

        Returns
        -------
        dict
            Dictionary of triangle neighbors
            
        """
        tri_nbr_list = self.DT.neighbors.tolist()
        tri_nbr_list = [[ele for ele in sub if ele != -1] for sub in tri_nbr_list]
        return dict(enumerate(tri_nbr_list))


    def compute_normals(self):
        """Compute surface normals for each triangle in mesh

        Returns
        -------
        normals : np.array (n_pts x 3)
            Dictionary of triangle neighbors
            
        """
        T = self.P[self.DT.simplices]
        U = T[:,2,:] - T[:,0,:]  # v3 - v1
        V = T[:,1,:] - T[:,0,:]  # v2 - v1
        normals = np.cross(U,V)
        normals /= np.linalg.norm(normals, axis=1)[:,None]
        return normals


