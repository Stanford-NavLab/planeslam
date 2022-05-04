"""Utilities for working with point cloud meshes

"""

import numpy as np
from scipy.spatial import Delaunay

import planeslam.general as general
import planeslam.geometry as geometry


def vertex_neighbors(vertex, mesh):
    """Retrieve vertex neighbors in mesh

    Parameters
    ----------
    vertex : int
        Vertex ID (index)
    mesh : scipy.spatial.Delaunay
        Mesh data structure 

    Returns
    -------
    list
        List of neighbor vertex IDs

    """
    idx_ptrs, idxs = mesh.vertex_neighbor_vertices
    return idxs[idx_ptrs[vertex]:idx_ptrs[vertex+1]]


def lidar_mesh(P):
    """Create a mesh from an unorganized LiDAR point cloud using Delaunay triangulation

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Point cloud points

    Returns
    -------
    mesh : scipy.spatial.Delaunay
        Mesh data structure 

    """
    # Map points to 2D (with inverse spherical projection)
    # TODO: handle wrapping
    thetas = np.arctan2(P[:,1], P[:,0])
    Rxy = np.sqrt(P[:,0]**2 + P[:,1]**2)
    phis = np.arctan2(P[:,2], Rxy)

    # Generate Delaunay triangulation
    return Delaunay(np.stack((thetas,phis), axis=1))


def prune_mesh(P, mesh, edge_len_lim):
    """Prune mesh by removing triangles with edge length exceeding specified length

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Point cloud points
    mesh : scipy.spatial.Delaunay
        Mesh data structure 
    edge_len_lim : float
        Maximum edge length to retain

    Returns
    -------
    mesh : scipy.spatial.Delaunay
        Pruned mesh data structure 

    """
    T = P[mesh.simplices]
    S1 = np.linalg.norm(T[:,0,:] - T[:,1,:], axis=1)  # side 1 lengths
    S2 = np.linalg.norm(T[:,1,:] - T[:,2,:], axis=1)  # side 2 lengths
    S3 = np.linalg.norm(T[:,2,:] - T[:,0,:], axis=1)  # side 3 lengths
    keep_idx_mask = (S1 < edge_len_lim) & (S2 < edge_len_lim) & (S3 < edge_len_lim) 

    # Prune the simplices
    mesh.simplices = mesh.simplices[keep_idx_mask]

    # Update other fields of tri data stucture
    mesh.equations = mesh.equations[keep_idx_mask]
    
    # Remap indices for neighbors
    full_idxs = np.arange(len(keep_idx_mask))
    keep_idxs = full_idxs[keep_idx_mask]
    discard_idxs = full_idxs[~keep_idx_mask]
    if len(keep_idxs) < len(keep_idx_mask):
        # NOTE: some reason this breaks when we transform to ENU??
        # Remap discard idxs to -1
        mesh.neighbors = general.remap(mesh.neighbors, discard_idxs, -np.ones(len(discard_idxs)))
        # Remap keep idxs to start at 0
        mesh.neighbors = general.remap(mesh.neighbors, keep_idxs, np.arange(len(keep_idxs)))
    mesh.neighbors = mesh.neighbors[keep_idx_mask]

    return mesh


def cluster_mesh_graph_search(P, mesh, normal_match_thresh=0.17, min_cluster_size=5):
    """Cluster mesh with graph search
    
    Parameters
    ----------
    P : np.array (n_pts x 3)
        Point cloud points
    mesh : scipy.spatial.Delaunay
        Mesh data structure
    normal_match_thresh : float
        Norm difference threshold to cluster triangles together
    min_cluster_size : int
        Minimum cluster size

    Returns
    -------
    clusters : list of lists
        List of triangle indices grouped into clusters
    avg_normals : list of np.array
        Average normal vectors for each cluster

    """
    # Compute surface normals
    T = P[mesh.simplices]
    U = T[:,2,:] - T[:,0,:]
    V = T[:,1,:] - T[:,0,:]
    normals = np.cross(U,V)
    normals /= np.linalg.norm(normals, axis=1)[:,None]

    # Create triangle neighbors dictionary
    tri_nbr_dict = create_tri_nbr_dict(mesh)

    # Graph search
    clusters = []  # clusters are idxs of triangles, triangles are idxs of points
    avg_normals = []
    to_cluster = set(range(len(mesh.simplices)))

    while to_cluster:
        root = to_cluster.pop()
        avg_normal = normals[root,:]

        cluster = [root]
        search_queue = set(tri_nbr_dict[root])
        search_queue = set([x for x in search_queue if x in to_cluster])  # don't search nodes that have already been clustered

        while search_queue:
            i = search_queue.pop()
            if np.linalg.norm(normals[i,:] - avg_normal) < normal_match_thresh:
                # Add node to cluster and remove from to_cluster
                cluster.append(i)
                to_cluster.remove(i)
                # Add its neighbors (that are not already clustered or search queue) to the search queue
                search_nbrs = tri_nbr_dict[i].copy()
                search_nbrs = [x for x in search_nbrs if x in to_cluster]
                search_nbrs = [x for x in search_nbrs if not x in search_queue]
                search_queue.update(search_nbrs)
                # Update average normal
                avg_normal = np.mean(normals[cluster], axis=0)
                avg_normal = avg_normal / np.linalg.norm(avg_normal)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
            avg_normals.append(avg_normal)

    return clusters, avg_normals


def find_cluster_boundary(cluster, tri_nbr_dict, mesh):
    """Find boundary vertices in cluster of triangles

    Parameters
    ----------
    cluster : list
        List of triangle indices denoting cluster
    tri_nbr_dict : dict
        Dictionary holding neighbor triangle indices for each triangle
    mesh : scipy.spatial.Delaunay
        Mesh data structure

    Returns
    -------
    bd_verts : set
        Set containing indices of vertices on boundary of cluster
        
    """
    bd_verts = set()  
    for tri_idx in cluster:
        tri_nbrs = set(tri_nbr_dict[tri_idx]) & set(cluster)
        if len(tri_nbrs) == 2:
            # 2 vertices not shared by neighbors are boundary points
            nbr_verts = mesh.simplices[list(tri_nbrs),:]
            vals, counts = np.unique(nbr_verts, return_counts=True)
            bd_nbr_verts = set(mesh.simplices[tri_idx,:])
            if 2 in counts:
                bd_nbr_verts.remove(vals[counts==2][0])
            bd_verts.update(bd_nbr_verts)
        elif len(tri_nbrs) == 1:
            # All 3 vertices are boundary points
            bd_verts.update(mesh.simplices[tri_idx])
        
    return bd_verts


def create_tri_nbr_dict(mesh):
    """Create dictionary storing triangle neighbors

    Parameters
    ----------
    mesh : scipy.spatial.Delaunay
        Mesh data structure

    Returns
    -------
    dict
        Dictionary of triangle neighbors
        
    """
    tri_nbr_list = mesh.neighbors.tolist()
    tri_nbr_list = [[ele for ele in sub if ele != None] for sub in tri_nbr_list]
    return dict(enumerate(tri_nbr_list))