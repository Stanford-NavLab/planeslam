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
    mesh.neighbors = mesh.neighbors[keep_idx_mask]
    # Remap indices for neighbors
    full_idxs = np.arange(len(keep_idx_mask))
    keep_idxs = full_idxs[keep_idx_mask]
    discard_idxs = full_idxs[~keep_idx_mask]
    if len(keep_idxs) < len(keep_idx_mask):
        # Remap discard idxs to -1
        mesh.neighbors = general.remap(mesh.neighbors, discard_idxs, -np.ones(len(discard_idxs)))
        # Remap keep idxs to start at 0
        mesh.neighbors = general.remap(mesh.neighbors, keep_idxs, np.arange(len(keep_idxs)))

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


def bd_plane_from_pts(pts, n):
    """Extract bounding rectangular plane from set of 3D points

    Given a set of cluster boundary points, project the points to the cluster normal 
    plane, then find the planar bounding box of the points.

    Parameters
    ----------
    pts : np.array (n_pts x 3)
        Cluster points
    n : np.array (3 x 1)
        Cluster normal vector

    Returns
    -------
    plane_pts : np.array (4 x 3)
        Planar bounding box points
        
    """
    plane_pts = np.empty((4,3))

    # Project to nearest cardinal plane to find bounding box points
    plane_idx = np.argsort(np.linalg.norm(np.eye(3) - np.abs(n), axis=0))[0]
    plane = np.eye(3)[:,plane_idx][:,None]
    pts_proj = geometry.project_points_to_plane(pts, plane)

    # Find 2D bounding box of points within plane
    # This should order the points counterclockwise starting from (-,-) point
    # TODO: store points as CW or CCW depending on direction of normal
    idx_count = 0
    for k in range(3):
        if k == plane_idx:
            plane_pts[:,k] = pts_proj[0,plane_idx]
        else:
            min = np.amin(pts_proj[:,k])
            max = np.amax(pts_proj[:,k])
            if idx_count == 0:
                plane_pts[:,k] = np.array([min, max, max, min])
            elif idx_count == 1:
                plane_pts[:,k] = np.array([min, min, max, max])
            idx_count += 1

    # Project back to original normal plane
    plane_pts = geometry.project_points_to_plane(plane_pts, n)
    
    return plane_pts


def scan_from_clusters(P, mesh, clusters, avg_normals, vertex_merge_thresh=1.0):
    """Convert clustered points to network of planes

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Point cloud points
    mesh : scipy.spatial.Delaunay
        Mesh data structure
    clusters : list of lists
        Point indices grouped into clusters (based on surface normals and locality)
    avg_normals : list of np.array (3 x 1)
        Average normal vector for each cluster of points
    vertex_merge_thresh : float
        Distance between vertices in order to merge them

    Returns
    -------
    vertices : np.array (n_verts x 3)
        Ordered array of vertices in scan
    faces : list of lists
        Sets of 4 vertex indices which form a face
    normals : np.array (n_faces x 3)
        Normal vectors for each face

    """
    # TODO: merge planes which are inside other planes into each other

    vertices = []
    faces = []  # sets of 4 vertex indices
    normals = []
    vertex_counter = 0

    # Sort clusters from largest to smallest
    cluster_sort_idx = np.argsort([-len(c) for c in clusters])
    clusters[:] = [clusters[i] for i in cluster_sort_idx]
    avg_normals[:] = [avg_normals[i] for i in cluster_sort_idx]

    for cluster_idx in range(len(clusters)):  
        c = clusters[cluster_idx]
        n = avg_normals[cluster_idx][:,None]
        cluster_pts_idxs = np.unique(mesh.simplices[c,:]) 
        cluster_pts = P[cluster_pts_idxs,:]
        # Extract bounding plane
        plane_pts = bd_plane_from_pts(cluster_pts, n)

        new_face = 4*[None]
        keep_mask = 4*[True]
        
        # Check if this plane shares any vertices with previous planes
        for i in range(len(vertices)):
            dists = np.linalg.norm(plane_pts - vertices[i], axis=1)
            best_match = np.argsort(dists)[0]
            if dists[best_match] < vertex_merge_thresh:
                new_face[best_match] = i
                keep_mask[best_match] = False
        
        for i in range(4):
            if new_face[i] is None:
                new_face[i] = vertex_counter
                vertex_counter += 1

        vertices += list(plane_pts[keep_mask,:])
        faces.append(new_face)
        normals.append(n)

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    normals = np.asarray(normals)
    
    return vertices, faces, normals