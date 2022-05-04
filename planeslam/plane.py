"""BoundedPlane class and utilities

This module defines the BoundedPlane class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt

from planeslam.general import normalize


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
        # TODO: check that vertices are coplanar and form a rectangle
        self.vertices = vertices

        # Form the basis vectors
        basis_x = normalize(vertices[1,:] - vertices[0,:])  # x is v2 - v1 
        basis_y = normalize(vertices[3,:] - vertices[0,:])  # y is v4 - v1 
        basis_z = np.cross(basis_x, basis_y)  # z is x cross y
        self.basis = np.vstack((basis_x, basis_y, basis_z)).T

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
        # TODO: transform basis and normal
        self.normal = R @ self.normal
        self.basis = R @ self.basis  # NOTE: is this right?
        self.center += t
    

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