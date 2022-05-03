"""Plane extraction from clustered points

"""

import numpy as np

import planeslam.geometry as geometry


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


def scan_from_pcl_clusters(P, clusters, normals_arr, vertex_merge_thresh=1.0):
    """Convert points clustered using PCL to a planescan

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Point cloud points
    clusters : PointIndices
        Point indices grouped into clusters using PCL region growing
    normals_arr : np.array (n_pts x 3)
        Array of normal vectors for each point
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

    for i, c in enumerate(clusters):
        idx = c.indices  
        cluster_pts = P[idx,:]

        # Compute average normal
        n = np.mean(normals_arr[idx,:], axis=0)[:,None]

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