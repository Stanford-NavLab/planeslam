"""BoundedPlane class and utilities

This module defines the BoundedPlane class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt

from planeslam.general import normalize
from planeslam.geometry.util import vector_projection
from planeslam.geometry.box import Box


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
            "Side lengths from vertices given to BoundedPlane constructor are not equal"

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


