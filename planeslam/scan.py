"""Scan class and utilities

This module defines the Scan class and relevant utilities.

"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import open3d as o3d

from planeslam.general import downsample
from planeslam.extraction import scan_from_clusters, planes_from_clusters
from planeslam.clustering import cluster_mesh_graph_search
from planeslam.mesh import LidarMesh
from planeslam.geometry.plane import BoundedPlane, plane_to_plane_dist, merge_plane
from planeslam.geometry.box import Box, box_from_pts
from planeslam.geometry.rectangle import Rectangle


class Scan:
    """Scan class.

    This class represents a processed LiDAR point cloud, in which the points have been
    clustered and converted to planes. 

    Attributes
    ----------
    planes : list
        List of BoundedPlane objects
    vertices : np.array (n_verts x 3)
        Ordered array of vertices in scan
    faces : list of lists
        Sets of 4 vertex indices which form a face
    center : np.array (3 x 1)
        Point at which scan is centered at (i.e. LiDAR pose position)
    
    Methods
    -------
    plot()

    """

    def __init__(self, planes, vertices=None, faces=None, center=None):
        """Constructor
        
        Parameters
        ----------
        planes : list
            List of BoundedPlane objects

        """
        self.planes = planes

        if vertices is not None:
            self.vertices = vertices
            self.faces = faces
        else:
            # TODO: get vertices and faces from planes
            pass

        if center is not None:
            self.center = center
        else:
            self.center = np.zeros((3,1))  # Default centered at 0
        

    def transform(self, R, t):
        """Transform scan by rotation R and translation t

        Parameters
        ----------
        R : np.array (3 x 3)
            Rotation matrix
        t : np.array (1 x 3)
            Translation vector
        
        """
        for p in self.planes:
            p.transform(R, t)
        # TODO: comment back in once vertex and face generation implemented
        # self.vertices = (R @ self.vertices.T).T + t
        # self.center += t[:,None]
    

    def plot_trace(self, show_normals=False, normal_scale=5):
        """Generate plotly plot trace

        TODO: sometimes plane mesh is not plotted properly, may be due to ordering of vertices

        Returns
        -------
        data : list
            List of graph objects to plot for scan

        """
        data = []
        colors = px.colors.qualitative.Plotly
        for i, p in enumerate(self.planes):
            data += p.plot_trace(name=str(i), color=colors[i%len(colors)])

        # Plot normal vectors
        if show_normals:
            n = len(self.planes)
            xs = [None for i in range(3*n)]
            ys = [None for i in range(3*n)]
            zs = [None for i in range(3*n)]
            for i, p in enumerate(self.planes):
                xs[3*i] = p.center[0]
                xs[3*i+1] = p.center[0] + normal_scale * p.normal.flatten()[0]
                ys[3*i] = p.center[1]
                ys[3*i+1] = p.center[1] + normal_scale * p.normal.flatten()[1]
                zs[3*i] = p.center[2]
                zs[3*i+1] = p.center[2] + normal_scale * p.normal.flatten()[2]
            data.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="red",width=2), showlegend=False))
        
        return data


    def o3d_geometries(self):
        """Generate open3d geometries for plotting
        
        """
        geoms = []
        for p in self.planes:
            verts = p.vertices
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(verts)
            line_set.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0]])
            mesh1 = o3d.geometry.TriangleMesh()
            mesh1.vertices = o3d.utility.Vector3dVector(verts)
            mesh1.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
            mesh2 = o3d.geometry.TriangleMesh()
            mesh2.vertices = o3d.utility.Vector3dVector(verts)
            mesh2.triangles = o3d.utility.Vector3iVector([[0,2,1],[0,3,2]])
            #mesh.paint_uniform_color([1, 0.706, 0])
            geoms += [line_set, mesh1, mesh2]

        return geoms

    
    def merge(self, scan, norm_thresh=0.1, dist_thresh=5.0):
        """Merge 
        
        Merge own set of planes (P) with set of planes Q.

        Parameters
        ----------
        scan : Scan
            Scan object to merge with
        norm_thresh : float 
            Correspodence threshold for comparing normal vectors
        dist_thesh : float
            Correspondence threshold for plane to plane distance
        
        Returns
        -------
        Scan
            Merged scan
        
        """
        P = self.planes
        Q = scan.planes
        merged_planes = []

        # Keep track of which faces in each scan have been matched
        P_unmatched = []
        Q_matched = []  

        for i, p in enumerate(P):
            # Compute projection of p onto it's own basis
            p_proj = (np.linalg.inv(p.basis) @ p.vertices.T).T
            merge_verts_2D = p_proj[:,0:2] 
            p_rect = Rectangle(p_proj[:,0:2])

            for j, q in enumerate(Q): 
                # Check if 2 planes are approximately coplanar
                if np.linalg.norm(p.normal - q.normal) < norm_thresh:
                    # Check plane to plane distance    
                    if plane_to_plane_dist(p, q) < dist_thresh:
                        # Project q onto p's basis
                        q_proj = (np.linalg.inv(p.basis) @ q.vertices.T).T
                        # Check overlap
                        q_rect = Rectangle(q_proj[:,0:2])
                        if p_rect.is_intersecting(q_rect):
                            # Add q to the correspondences set
                            merge_verts_2D = np.vstack((merge_verts_2D, q_proj[:,0:2]))
                            Q_matched.append(j)
            
            if len(merge_verts_2D) > 4:
                # Merge vertices using 2D bounding box
                merge_box = box_from_pts(merge_verts_2D)
                # Project back into 3D
                merge_verts = np.hstack((merge_box.vertices(), np.tile(p_proj[0,2], (4,1))))
                merge_verts = (p.basis @ merge_verts.T).T
                merged_planes.append(BoundedPlane(merge_verts))
            else:
                # Mark this plane as unmatched
                P_unmatched.append(i)
        
        # Add unmatched planes to merged set
        for i in P_unmatched:
            merged_planes.append(P[i])
        Q_unmatched = set(range(len(Q)))
        Q_unmatched.difference_update(Q_matched)
        for i in Q_unmatched:
            merged_planes.append(Q[i])

        return Scan(merged_planes)


    def reduce_inside(self):
        """Reduce by checking for planes inside of each other
        
        Reduce scan by merging vertices and planes.

        Merge planes inside of each other:
            Iterate through the planes:
                For each plane P:
                    Project other plane Q to P's basis
                    Check if Q is fully contained within P
                        If so, get rid of Q 
        
        Merge planes next to each other:
            Check if there is a pair of vertices close to another pair of vertices
            for another plane with similar normal


        """
        # Iterate through the planes, and for each new plane, check if 
        # any of it's vertices are close to any existing vertices

        # check = set(range(len(self.planes)))
        # keep = set()

        # while check:
        #     i = check.pop()

        keep = set(range(len(self.planes)))

        for i, p in enumerate(self.planes):
            if i in keep:
                p_proj = (np.linalg.inv(p.basis) @ p.vertices.T).T
                p_rect = Rectangle(p_proj[:,0:2])
                to_remove = set()
                for j in keep:
                    if j == i:
                        continue
                    q = self.planes[j]
                    if np.dot(p.normal.T, q.normal) > 0.866:
                        q_proj = (np.linalg.inv(p.basis) @ q.vertices.T).T
                        q_rect = Rectangle(q_proj[:,0:2])
                        if p_rect.contains(q_rect):
                            print(f'{i} contains {j}')
                            to_remove.add(j)

            keep.difference_update(to_remove)
        
        keep = list(keep)
        self.planes = [self.planes[i] for i in keep]

    
    def reduce_intersections():
        """Reduce by checking intersections
        
        """
        

    def remove_small_planes(self, area_thresh=1.0):
        """Remove planes with small area
        
        """
        keep = list(range(len(self.planes)))

        for i, p in enumerate(self.planes):
            # Project p into it's own basis
            p_proj = (np.linalg.inv(p.basis) @ p.vertices.T).T
            # Form 2D box from projection
            p_box = Box(p_proj[0,0:2], p_proj[2,0:2])
            if p_box.area() < area_thresh:
                keep.remove(i)
        
        self.planes = [self.planes[i] for i in keep]


    
    def fuse_edges(self, vertex_merge_thresh=2.0):
        """Fuse edges
        
        """
        # # Get list of all edges
        # E = []
        # for p in self.planes:
        #     E += p.edges()
        
        # # Edge distance metric
        # def edge_dist(e1, e2):
        #     return min(np.linalg.norm(e1[0] - e2[0]) + np.linalg.norm(e1[1] - e2[1]),
        #         np.linalg.norm(e1[1] - e2[0]) + np.linalg.norm(e1[0] - e2[1]))

        # num_edges = len(E)
        # DM = np.zeros((num_edges,num_edges))  # distance matrix
        # for i in range(num_edges):
        #     for j in range(num_edges):
        #         DM[i,j] = edge_dist(E[i], E[j])

        # NOTE: might want to resort planes by size after each merge
        # Use vertices of first plane as initial anchors
        vertices = list(self.planes[0].vertices)
        update_idxs = []
        update_planes = []

        # Iterate over remaining planes
        for i, p in enumerate(self.planes[1:]):
            plane_pts = p.vertices
            new_face = -np.ones(4, dtype=int)
            merge_mask = np.zeros(4, dtype=bool)

            # Check if this plane shares any vertices with previous planes
            for k in range(len(vertices)):
                dists = np.linalg.norm(plane_pts - vertices[k], axis=1)
                best_match = np.argsort(dists)[0]
                if dists[best_match] < vertex_merge_thresh:
                    new_face[best_match] = k
                    merge_mask[best_match] = True

            # If shared, adjust plane accordingly
            if sum(merge_mask) == 2:
                anchor_idxs = new_face[new_face!=-1]
                anchor_verts = np.asarray(vertices)[anchor_idxs]
                new_plane = merge_plane(merge_mask, anchor_verts, plane_pts, p.normal)

                vertices += list(new_plane[~merge_mask,:])
                update_idxs.append(i+1)
                update_planes.append(BoundedPlane(new_plane))
            else:
                vertices += list(plane_pts)

        # Update planes
        for i, idx in enumerate(update_idxs):
            self.planes[idx] = update_planes[i]



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
    P = downsample(P, factor=2, axis=0)

    # Create the mesh
    mesh = LidarMesh(P)
    # Prune the mesh
    mesh.prune(edge_len_lim=10)
    # Cluster the mesh with graph search
    clusters, avg_normals = cluster_mesh_graph_search(mesh)

    # Form scan topology
    # planes, vertices, faces = scan_from_clusters(mesh, clusters, avg_normals)
    # return Scan(planes, vertices, faces)
    planes = planes_from_clusters(mesh, clusters, avg_normals)
    return Scan(planes)


def pc_extraction(P):
    """Point cloud to scan

    Parameters
    ----------
    P : np.array (n_pts x 3)
        Unorganized point cloud

    Returns
    -------
    mesh : LidarMesh
    clusters : 
    Scan 
        Scan representing input point cloud
    
    """
    # Downsample
    P = downsample(P, factor=2, axis=0)

    # Create the mesh
    mesh = LidarMesh(P)
    # Prune the mesh
    mesh.prune(edge_len_lim=10)
    # Cluster the mesh with graph search
    clusters, avg_normals = cluster_mesh_graph_search(mesh)

    # Form scan topology
    planes, vertices, faces = scan_from_clusters(mesh, clusters, avg_normals)
    return mesh, clusters, Scan(planes, vertices, faces)
