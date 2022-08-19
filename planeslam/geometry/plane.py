"""BoundedPlane class and utilities

This module defines the BoundedPlane class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from planeslam.general import normalize
from planeslam.geometry.util import vector_projection
from planeslam.geometry.box import Box, box_from_pts
from planeslam.planning.zonotope import Zonotope


class BoundedPlane:
    """Bounded Plane class.

    This class represents a rectangularly bounded plane in 3D, represented by 4 coplanar 
    vertices which form a rectangle. 

    Attributes
    ----------
    vertices : np.array (4 x 3)
        Ordered array of vertices 
    normal : np.array (3 x 1)
        Normal vector
    basis : np.array (3 x 3)
        x,y,z basis column vectors: z is normal, and x and y span the plane space
    center : np.array (3 x 1)
        Center of the plane
    
    Methods
    -------
    plot()

    """

    def __init__(self, vertices):
        """Constructor
        
        Parameters
        ----------
        vertices : np.array (4 x 3)
            Ordered array of vertices 

        """
        # Check side lengths
        S = np.diff(vertices, axis=0, append=vertices[0][None,:])  # Side vectors
        assert np.all(np.isclose(S[0], -S[2])) and np.all(np.isclose(S[1], -S[3])), \
            "Side lengths from vertices given to BoundedPlane constructor are not equal "

        # Form the basis vectors
        basis_x = normalize(S[0])  # x is v2 - v1 
        basis_y = normalize(-S[3])  # y is v4 - v1 
        basis_z = np.cross(basis_x, basis_y)  # z is x cross y
        self.basis = np.vstack((basis_x, basis_y, basis_z)).T

        # Coplanarity check
        assert np.all(np.isclose(np.cross(basis_z, np.cross(-S[1], S[2])), 0)), \
            "Vertices given to BoundedPlane constructor are not coplanar"

        self.vertices = vertices
        self.normal = basis_z[:,None]  # Normal is z
        self.center = np.mean(vertices, axis=0)
        

    def transform(self, R, t):
        """Transform plane by rotation R and translation t

        Parameters
        ----------
        R : np.array (3 x 3)
            Rotation matrix
        t : np.array (1 x 3)
            Translation vector
        
        """
        self.vertices = (R @ self.vertices.T).T + t
        self.basis = R @ self.basis  # NOTE: is this right?
        self.normal = self.basis[:,2][:,None]
        self.center = R @ self.center + t

    
    def to_2D(self):
        """Projection of a 3D plane to 2D
        
        Compute projection of a (rectangularly bounded) plane that exists in 3D  
        into its 2D subspace based on its basis vectors.

        Returns
        -------
        box_2D : Box
            2D box 
        z : float
            Z coordinate of box

        """
        # Change basis to plane's basis
        A = np.linalg.inv(self.basis)
        V = (A @ self.vertices.T).T
        V_2D = V[:,0:2]
        box_2D = Box(np.amin(V_2D, axis=0), np.amax(V_2D, axis=0))
        z = V[0,2]
        return box_2D, z


    def edges(self):
        """Return edges of planes
        
        Returns
        -------
        list of tuples
            List of edges as (v_i,v_j) vertex tuples 

        """
        V = self.vertices
        return [(V[i], V[(i+1)%4]) for i in range(4)]


    def area(self):
        """Compute the area of this plane
        
        Returns
        -------
        float
            Area of plane

        """
        # Project onto own basis
        p_proj = (np.linalg.inv(self.basis) @ self.vertices.T).T
        # Form 2D box from projection
        p_box = Box(p_proj[0,0:2], p_proj[2,0:2])
        return p_box.area()


    def to_zonotope(self):
        """Convert this plane to zonotope representation
        
        Returns
        -------
        Zonotope
            Zonotope representation of plane

        """
        c = self.center[:,None]
        G = np.diff(self.vertices[:3], axis=0).T / 2
        return Zonotope(c, G)
    

    def dist_to_point(self, x):
        """Compute shortest distance from plane to 3D point

        Distance is positive if in the direction of the plane normal, negative
        if in the opposite direction.
        
        Parameters
        ----------
        x : np.array (3)
            Point to get distance to

        Returns
        -------
        float
            Point to plane distance

        """
        return -self.normal.T @ (self.center - x)


    def check_line_intersect(self, line):
        """Check if line segment intersects plane
        
        Parameters
        ----------
        line : np.array (2 x 3)
            Line segment defined by two points
        
        Returns
        -------
        bool
            True if intersecting, False otherwise
        
        """
        # Project to plane basis
        plane_proj = self.vertices @ self.basis
        plane_proj_z = plane_proj[0,2]
        line_proj = line @ self.basis

        # Find intersection point of infinite line with infinite plane
        # by setting z for line (since z is constant for projected plane)
        line_x0 = line_proj[0,:]
        line_v = line_proj[1,:] - line_proj[0,:]

        if line_v[2] == 0:
            return False
        else:
            c = (plane_proj_z - line_x0[2]) / line_v[2]
            intersection_pt = line_x0 + c * line_v

            # Check if intersection point lies within line segment and within plane boundaries
            in_line = c < 1 and c > 0
            in_plane = box_from_pts(plane_proj[:,:2]).contains(intersection_pt[:2])
            return in_line and in_plane
    

    def plot(self, ax=None, color='b', show_normal=False):
        """Plot

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on, if not provided, will generate new set of axes
        color : optional
            Color to plot, default blue
        show_normal : bool, optional
            Whether to plot normal vector
        
        """
        if ax == None:
            fig, ax = plt.subplots()
        
        V = self.vertices
        ax.plot(np.hstack((V[:,0],V[0,0])), np.hstack((V[:,1],V[0,1])), np.hstack((V[:,2],V[0,2])), color=color)

        if show_normal:
            c = self.center
            n = 10 * self.normal  # TODO: quiver scaling is currently arbitrary
            ax.quiver(c[0], c[1], c[2], n[0], n[1], n[2], color=color)
            # Plot basis
            x = 10 * self.basis[:,0]
            y = 10 * self.basis[:,1]
            ax.quiver(c[0], c[1], c[2], x[0], x[1], x[2], color=color)
            ax.quiver(c[0], c[1], c[2], y[0], y[1], y[2], color=color)

    
    def plot_trace(self, name=None, color=None):
        """Generates plotly trace for this plane to be plotted
        
        """
        V = self.vertices
        mesh = go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=[0,0], j=[1,2], k=[2,3],
            opacity=0.5, showlegend=True, name=name, color=color)
        lines = go.Scatter3d(x=np.hstack((V[:,0],V[0,0])), y=np.hstack((V[:,1],V[0,1])), z=np.hstack((V[:,2],V[0,2])), 
            mode='lines', line=dict(color= 'rgb(70,70,70)', width=1), showlegend=False)
        plane_data = [mesh, lines]
        return plane_data

        
    
def plane_to_plane_dist(plane_1, plane_2):
    """Plane-to-Plane distance
    
    Shortest distance between two (rectangularly bounded) planes. Computed
    by taking the centroid to centroid vector, and projecting it along the
    average normal of the two planes, and taking the norm of the projected
    vector. Meant for planes with close normals.

    Parameters
    ----------
    plane_1 : BoundedPlane
        Rectangularly bounded plane represented by 4 vertices
    plane_2 : BoundedPlane
        Rectangularly bounded plane represented by 4 vertices
    
    Returns
    -------
    float
        Plane-to-plane distance
    
    """
    c2c_vector = plane_1.center - plane_2.center  # NOTE: may be issue with c2c_vector pointing opposite to avg_normal
    avg_normal = (plane_1.normal + plane_2.normal) / 2
    return np.linalg.norm(vector_projection(c2c_vector, avg_normal))


def merge_plane(mask, anchor_verts, old_verts, normal):
    """Merge plane
    
    Merge a plane with another using 2 anchor vertices.

    Mask indicates which vertices from old_vertices are being merged with
    the anchors (and also which vertices in the plane the anchors belong to)

    [True, True, False, False] = v2 - v1 = x
    [False, True, True, False] = v3 - v2 = y
    [False, False, True, True] = v4 - v3 = -x
    [True, False, False, True] = v4 - v1 = y

    Parameters
    ----------

    Returns
    -------

    """
    new_verts = np.empty((4,3))

    if np.all(mask == [True, True, False, False]):
        basis_x = normalize(np.diff(anchor_verts, axis=0))
        basis_y = np.cross(normal.T, basis_x)
        S2_len = np.linalg.norm(old_verts[2] - old_verts[1])
        v3_new = anchor_verts[1] + S2_len * basis_y
        v4_new = anchor_verts[0] + S2_len * basis_y
        new_verts[2] = v3_new
        new_verts[3] = v4_new
        new_verts[mask] = anchor_verts

    elif np.all(mask == [False, True, True, False]):
        basis_y = normalize(np.diff(anchor_verts, axis=0))
        basis_x = np.cross(basis_y, normal.T)
        S1_len = np.linalg.norm(old_verts[1] - old_verts[0])
        v4_new = anchor_verts[1] - S1_len * basis_x
        v1_new = anchor_verts[0] - S1_len * basis_x
        new_verts[0] = v1_new
        new_verts[3] = v4_new
        new_verts[mask] = anchor_verts

    elif np.all(mask == [False, False, True, True]):
        basis_x = -normalize(np.diff(anchor_verts, axis=0))
        basis_y = np.cross(normal.T, basis_x)
        S2_len = np.linalg.norm(old_verts[2] - old_verts[1])
        v2_new = anchor_verts[0] - S2_len * basis_y
        v1_new = anchor_verts[1] - S2_len * basis_y
        new_verts[0] = v1_new
        new_verts[1] = v2_new
        new_verts[mask] = anchor_verts
    
    elif np.all(mask == [True, False, False, True]):
        basis_y = normalize(np.diff(anchor_verts, axis=0))
        basis_x = np.cross(basis_y, normal.T)
        S1_len = np.linalg.norm(old_verts[1] - old_verts[0])
        v3_new = anchor_verts[1] + S1_len * basis_x
        v2_new = anchor_verts[0] + S1_len * basis_x
        new_verts[2] = v3_new
        new_verts[1] = v2_new
        new_verts[mask] = anchor_verts

    else:
        print(mask, ", is not a valid mask")
    
    return new_verts


def order_plane_verts(V, n):
    """Order plane vertices based on normal

    Orders points counterclockwise with respect to the normal (i.e. right hand rule)

    Parameters
    ----------
    V : np.array (4 x 3)
        Plane vertices
    n : np.array (3 x 1)
        Normal vector

    Returns
    -------
    V : np.array (4 x 3)
        Ordered plane vertices

    """
    # TODO: unfinished
    # Check if current ordering has normal in right direction

    # If not, swap ordering
