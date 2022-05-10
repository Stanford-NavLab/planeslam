"""PlaneSet class and utilities

This module defines the PlaneSet class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt

from planeslam.extraction import box_from_pts
from planeslam.plane import BoundedPlane, plane_to_plane_dist


class PlaneSet:
    """PlaneSet class.

    This class represents a set of BoundedPlane objects. 

    Attributes
    ----------
    planes : list
        List of BoundedPlane objects
    
    Methods
    -------
    plot()

    """

    def __init__(self, planes):
        """Constructor
        
        Parameters
        ----------
        planes : list
            List of BoundedPlane objects

        """
        self.planes = planes
        

    def transform(self, R, t):
        """Transform planeset by rotation R and translation t

        Parameters
        ----------
        R : np.array (3 x 3)
            Rotation matrix
        t : np.array (1 x 3)
            Translation vector
        
        """
        for p in self.planes:
            p.transform(R, t)
    

    def plot(self, ax=None, color='b', show_normals=False):
        """Plot

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on, if not provided, will generate new set of axes
        color : optional
            Color to plot, default blue
        show_normals : bool, optional
            Whether to plot normal vectors for each plane
        
        """
        if ax == None:
            fig, ax = plt.subplots()
        
        for p in self.planes:
            p.plot(ax, color, show_normals)
            
    
    def merge(self, Q, norm_thresh=0.1, dist_thresh=5.0):
        """Merge 
        
        Merge own set of planes (P) with set of planes Q

        Parameters
        ----------
        Q : PlaneSet
            PlaneSet Object
        norm_thresh : float 
            Correspodence threshold for comparing normal vectors
        dist_thesh : float
            Correspondence threshold for plane to plane distance
        
        Returns
        -------
        PlaneSet
            Merged planes
        
        """
        P = self.planes
        merged_planes = []

        # Keep track of which faces in each scan have been matched
        P_unmatched = []
        Q_matched = []  

        for i, p in enumerate(P):
            # Compute projection of p onto it's own basis
            p_proj = (np.linalg.inv(p.basis) @ p.vertices.T).T
            merge_verts_2D = p_proj[:,0:2] 

            for j, q in enumerate(Q): 
                # Check if 2 planes are approximately coplanar
                if np.linalg.norm(p.normal - q.normal) < norm_thresh:
                    # Check plane to plane distance    
                    if plane_to_plane_dist(p, q) < dist_thresh:
                        # NOTE: skip the overlap check for now - need to implement zonotopes and intersection
                        # Project q onto p's basis
                        q_proj = (np.linalg.inv(p.basis) @ q.vertices.T).T
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

        return PlaneSet(merged_planes)