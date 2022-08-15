"""LidarMesh class

This module defines the LidarMesh class and relevant utilities.

"""

import numpy as np
from scipy.spatial import Delaunay
import plotly.graph_objects as go
import open3d as o3d

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
        #self.tri_nbr_dict = self.create_tri_nbr_dict()
    

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


    def neighborhood(self, tri_idx, r=1):
        """Given a triangle T in the mesh, return the triangles in the neighborhood of T with range r (including T)
        
        Parameters
        ----------
        tri_idx : int
            Index of triangle to query neighborhood
        r : int
            Size of neighborhood in number of hops

        Returns
        -------
        list
            List of indices of triangles in the neighborhood

        """
        neighborhood = {tri_idx}
        for i in range(r):
            new_tris = set()
            for tri in neighborhood:
                new_tris.update(self.tri_nbr_dict[tri])
            neighborhood.update(new_tris)
        return list(neighborhood)


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


    def smooth_laplacian(self, iterations=3):
        """Smooth the mesh using Laplacian filter
        
        """
        # Create open3d mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(self.P)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(self.DT.simplices)

        smoothed_mesh = o3d_mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        self.P = np.asarray(smoothed_mesh.vertices)


    def plot_trace(self):
        """Generate plotly plot trace for mesh 

        Returns
        -------
        data : list
            go.Mesh3d and go.Scatter3d containing mesh and line data to plot

        """
        mesh_data = go.Mesh3d(x=self.P[:,0], y=self.P[:,1], z=self.P[:,2], 
            i=self.DT.simplices[:,0], j=self.DT.simplices[:,1], k=self.DT.simplices[:,2], flatshading=True, opacity=0.75)

        # Extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
        Xe = []; Ye = []; Ze = []
        for T in self.P[self.DT.simplices]:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])
            
        # Show mesh triangle sides
        lines = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', name='',
                        line=dict(color= 'rgb(70,70,70)', width=1))  

        data = [mesh_data, lines]
        return data