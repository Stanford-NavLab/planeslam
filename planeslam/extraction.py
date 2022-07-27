"""Plane extraction from clustered points

"""

import numpy as np
import scipy.spatial

from planeslam.general import downsample
from planeslam.mesh import LidarMesh
from planeslam.geometry.util import project_points_to_plane
from planeslam.geometry.plane import BoundedPlane, merge_plane
from planeslam.geometry.box import Box
from planeslam.clustering import sort_mesh_clusters, mesh_cluster_pts, cluster_mesh_graph_search


def oriented_bd_plane_from_pts(pts, n):
    """Extract bounding rectangular plane with arbitrary orientation from set of 3D points

    Given a set of cluster boundary points, project the points to the cluster normal 
    plane, then find the oriented planar bounding box of the points.

    Parameters
    ----------
    pts : np.array (n_pts x 3)
        Cluster points
    n : np.array (3 x 1)
        Cluster normal vector

    Returns
    -------
    plane_pts : np.array (4 x 3)
        Planar oriented bounding box points
        
    """
    plane_pts = np.empty((4,3))

    # Project to nearest cardinal plane to find bounding box points
    plane_idx = np.argsort(np.linalg.norm(np.eye(3) - np.abs(n), axis=0))[0]
    plane = np.eye(3)[:,plane_idx][:,None]
    pts_proj = project_points_to_plane(pts, plane)

    # -- Find oriented bounding box of points within the plane --

    # Find principal components of the points
    c = pts_proj.mean(axis=0)
    cov = np.sum([np.outer(pts_proj[i,:]-c,pts_proj[i,:]-c) for i in range(pts_proj.shape[0])],axis=0)
    (_,_,V) = np.linalg.svd(cov)
    v1 = V[0,:]
    v2 = V[1,:]

    # Get corners of the bounding box
    v1_dists = (pts_proj-c) @ v1
    v2_dists = (pts_proj-c) @ v2

    c1 = c + v1_dists.min() * v1 + v2_dists.min() * v2
    c2 = c + v1_dists.max() * v1 + v2_dists.min() * v2
    c3 = c + v1_dists.max() * v1 + v2_dists.max() * v2
    c4 = c + v1_dists.min() * v1 + v2_dists.max() * v2

    v1 = c2-c1
    v2 = c3-c2

    if np.cross(v1,v2) @ n > 0:
        plane_pts = np.vstack([c1,c2,c3,c4])
    else:
        plane_pts = np.vstack([c2,c1,c4,c3])

    return project_points_to_plane(plane_pts, n)


def bd_plane_from_pts_basis(pts, n, basis):
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

    # Project to basis
    pts_proj = pts @ np.linalg.inv(basis).T

    # Use normal to determine which dimensions to extract bounding box from
    plane_idx = np.argsort(np.linalg.norm(np.hstack((basis, -basis)) - n, axis=0))[0] % 3
    axes = {0,1,2}  # x,y,z
    axes.remove(plane_idx)
    axes = list(axes)

    # Find 2D bounding box of points within plane
    # Orders points counterclockwise with respect to the normal (i.e. right hand rule)
    plane_pts[:,plane_idx] = pts_proj[0,plane_idx]
    min = np.amin(pts_proj[:,axes], axis=0)
    max = np.amax(pts_proj[:,axes], axis=0)

    for i, ax in enumerate(axes):
        if i == 0:  # First coordinate
            if n[plane_idx] > 0:  # Positively oriented normal
                if plane_idx == 0 or plane_idx == 2:  # x or z normal
                    plane_pts[:,ax] = np.array([min[i], max[i], max[i], min[i]])  # Case 1
                else: # y normal
                    plane_pts[:,ax] = np.array([max[i], min[i], min[i], max[i]])  # Case 2
            else:  # Negatively oriented normal
                if plane_idx == 0 or plane_idx == 2:  # x or z normal
                    plane_pts[:,ax] = np.array([max[i], min[i], min[i], max[i]])  # Case 2
                else: # y normal
                    plane_pts[:,ax] = np.array([min[i], max[i], max[i], min[i]])  # Case 1
        else:  # Second coordinate
            plane_pts[:,ax] = np.array([min[i], min[i], max[i], max[i]])  # Case 3

    # Project back to standard basis
    plane_pts = plane_pts @ basis.T

    return plane_pts


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
    pts_proj = project_points_to_plane(pts, plane)

    # Find 2D bounding box of points within plane
    # Orders points counterclockwise with respect to the normal (i.e. right hand rule)
    plane_pts[:,plane_idx] = pts_proj[0,plane_idx]
    axes = {0,1,2}  # x,y,z
    axes.remove(plane_idx)
    axes = list(axes)
    min = np.amin(pts_proj[:,axes], axis=0)
    max = np.amax(pts_proj[:,axes], axis=0)

    for i, ax in enumerate(axes):
        if i == 0:  # First coordinate
            if n[plane_idx] > 0:  # Positively oriented normal
                if plane_idx == 0 or plane_idx == 2:  # x or z normal
                    plane_pts[:,ax] = np.array([min[i], max[i], max[i], min[i]])  # Case 1
                else: # y normal
                    plane_pts[:,ax] = np.array([max[i], min[i], min[i], max[i]])  # Case 2
            else:  # Negatively oriented normal
                if plane_idx == 0 or plane_idx == 2:  # x or z normal
                    plane_pts[:,ax] = np.array([max[i], min[i], min[i], max[i]])  # Case 2
                else: # y normal
                    plane_pts[:,ax] = np.array([min[i], max[i], max[i], min[i]])  # Case 1
        else:  # Second coordinate
            plane_pts[:,ax] = np.array([min[i], min[i], max[i], max[i]])  # Case 3

    # Project back to original normal plane
    plane_pts = project_points_to_plane(plane_pts, n)
    
    return plane_pts


def scan_from_clusters(mesh, clusters, avg_normals, vertex_merge_thresh=1.0):
    """Convert clustered points to network of planes

    Parameters
    ----------
    mesh : LidarMesh
        Mesh object for clusters
    clusters : list of lists
        Point indices grouped into clusters (based on surface normals and locality)
    avg_normals : list of np.array (3 x 1)
        Average normal vector for each cluster of points
    vertex_merge_thresh : float
        Distance between vertices in order to merge them

    Returns
    -------
    planes : list
        List of BoundedPlanes
    vertices : np.array (n_verts x 3)
        Ordered array of vertices in scan
    faces : np.array (n_faces, 4)
        Sets of 4 vertex indices which form a face

    """
    # TODO: merge planes which are inside other planes into each other

    vertices = []
    faces = []  
    planes = []
    vertex_counter = 0

    # Sort clusters from largest to smallest
    clusters, avg_normals = sort_mesh_clusters(clusters, avg_normals)

    # Find extraction basis based on normals
    basis = np.zeros((3,3))
    basis[:,2] = avg_normals[0]  # choose first cluster's normal as z
    dps = np.asarray(avg_normals) @ basis[:,2]
    orth_idxs = np.nonzero(np.abs(dps) < 0.2)[0]  # indices of normals approximately orthonormal to z
    basis[:,0] = avg_normals[orth_idxs[0]]  # choose the first one as x
    basis[:,1] = np.cross(basis[:,2], basis[:,0])

    for i, c in enumerate(clusters):  
        n = avg_normals[i][:,None]
        cluster_pts = mesh_cluster_pts(mesh, c)  # Extract points from cluster

        # Extract bounding plane
        plane_pts = bd_plane_from_pts_basis(cluster_pts, n, basis)
        new_face = -np.ones(4, dtype=int)  # New face indices
        merge_mask = np.zeros(4, dtype=bool)  # Which of the new plane points to merge with existing points
        
        # Check if this plane shares any vertices with previous planes
        for i in range(len(vertices)):
            dists = np.linalg.norm(plane_pts - vertices[i], axis=1)
            best_match = np.argsort(dists)[0]
            if dists[best_match] < vertex_merge_thresh:
                new_face[best_match] = i
                merge_mask[best_match] = True

        # If shared, adjust plane accordingly
        # TODO: handle other cases (sum(merge_mask) == 1,3,4)
        if sum(merge_mask) == 2:
            anchor_idxs = new_face[new_face!=-1]
            anchor_verts = np.asarray(vertices)[anchor_idxs]
            new_plane = merge_plane(merge_mask, anchor_verts, plane_pts, n)

            vertices += list(new_plane[~merge_mask,:])
            planes.append(BoundedPlane(new_plane))

        # Otherwise, add all new vertices
        else:
            vertices += list(plane_pts)
            planes.append(BoundedPlane(plane_pts))

        # Set new face indices
        for i in range(4):
            if new_face[i] == -1:
                new_face[i] = vertex_counter
                vertex_counter += 1

        faces.append(new_face)

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    
    return planes, vertices, faces


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

    planes = []
    vertices = []
    faces = []  # Sets of 4 vertex indices
    vertex_counter = 0

    for i, c in enumerate(clusters):
        idx = c.indices  
        cluster_pts = P[idx,:]

        # Compute average normal
        cluster_normals = normals_arr[idx,:]
        cluster_normals = cluster_normals[~np.isnan(cluster_normals[:,0]),:]
        n = np.mean(cluster_normals, axis=0)[:,None]

        # Extract bounding plane
        plane_pts = bd_plane_from_pts(cluster_pts, n)
        new_face = -np.ones(4, dtype=int)  # New face indices
        merge_mask = np.zeros(4, dtype=bool)  # Which of the new plane points to merge with existing points
        
        # Check if this plane shares any vertices with previous planes
        for i in range(len(vertices)):
            dists = np.linalg.norm(plane_pts - vertices[i], axis=1)
            best_match = np.argsort(dists)[0]
            if dists[best_match] < vertex_merge_thresh:
                new_face[best_match] = i
                merge_mask[best_match] = True

        # If shared, adjust plane accordingly
        if sum(merge_mask) > 0:
            if sum(merge_mask) == 2:
                anchor_idxs = new_face[new_face!=-1]
                anchor_verts = np.asarray(vertices)[anchor_idxs]
                new_plane = merge_plane(merge_mask, anchor_verts, plane_pts, n)

                vertices += list(new_plane[~merge_mask,:])
                planes.append(BoundedPlane(new_plane))

        # Otherwise, add all new vertices
        else:
            vertices += list(plane_pts)
            planes.append(BoundedPlane(plane_pts))

        # Set new face indices
        for i in range(4):
            if new_face[i] == -1:
                new_face[i] = vertex_counter
                vertex_counter += 1

        faces.append(new_face)

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    
    return planes, vertices, faces


def planes_from_clusters(mesh, clusters, avg_normals):
    """Convert clustered points to a set of planes

    Parameters
    ----------
    mesh : LidarMesh
        Mesh object for clusters
    clusters : list of lists
        Point indices grouped into clusters (based on surface normals and locality)
    avg_normals : list of np.array (3 x 1)
        Average normal vector for each cluster of points

    Returns
    -------
    planes : list of BoundedPlanes
        List of planes

    """ 
    planes = []

    # Sort clusters from largest to smallest
    clusters, avg_normals = sort_mesh_clusters(clusters, avg_normals)

    # Find extraction basis based on normals
    basis = np.zeros((3,3))
    basis[:,2] = avg_normals[0]  # choose first cluster's normal as z
    dps = np.asarray(avg_normals) @ basis[:,2]
    orth_idxs = np.nonzero(np.abs(dps) < 0.2)[0]  # indices of normals approximately orthonormal to z
    basis[:,0] = avg_normals[orth_idxs[0]]  # choose the first one as x
    basis[:,1] = np.cross(basis[:,2], basis[:,0])

    for i, c in enumerate(clusters):  
        n = avg_normals[i][:,None]
        cluster_pts = mesh_cluster_pts(mesh, c)  # Extract points from cluster

        # Extract bounding plane
        plane_pts = bd_plane_from_pts_basis(cluster_pts, n, basis)
        planes.append(BoundedPlane(plane_pts))
    
    return planes


def planes_from_pcl_clusters(P, clusters, normals_arr):
    """Convert clustered points to a set of planes

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Point cloud points
    clusters : PointIndices
        Point indices grouped into clusters using PCL region growing
    normals_arr : np.array (n_pts x 3)
        Array of normal vectors for each point

    Returns
    -------
    planes : list of BoundedPlanes
        List of planes

    """
    planes = []

    for c in clusters:  
        idx = c.indices  
        cluster_pts = P[idx,:]

        # Compute average normal
        n = np.mean(normals_arr[idx,:], axis=0)[:,None]
        
        # Extract bounding plane
        plane_pts = bd_plane_from_pts(cluster_pts, n)

        bplane = BoundedPlane(plane_pts)
        planes.append(bplane)
    
    return planes


def pc_to_planes(P):
    """Point cloud to planes

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
    P = downsample(P, factor=5, axis=0)

    # Create the mesh
    mesh = LidarMesh(P)
    # Prune the mesh
    mesh.prune(10)
    # Cluster the mesh with graph search
    clusters, avg_normals = cluster_mesh_graph_search(mesh)

    # Extract planes
    return planes_from_clusters(mesh, clusters, avg_normals)

def orient_normals(P,normals):
    """Orient normals to be consistently pointing towards the origin

    Parameters
    ---------

    Returns
    -------
    normals: np.array
    """
    dot = np.sum(P*normals,axis=1)
    idxs = np.where(dot > 0)
    normals[idxs] *= -1
    return normals

