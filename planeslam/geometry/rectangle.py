"""Rectangle class and utilities

This module defines the Rectangle class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


class Rectangle:
    """Rectangle class.

    This class represents a 2D rectangle defined by 4 vertices

    Attributes
    ----------
    vertices : np.array (4 x 2)
        Rectangle vertices
    
    Methods
    -------
    plot()

    """

    def __init__(self, vertices):
        """Constructor
        
        Parameters
        ----------
        vertices : np.array (4 x 2)
            Rectangle vertices, ordered counterclockwise starting from bottom left

        """
        # TODO: check if vertices form a rectangle and are ordered properly
        self.vertices = vertices
        

    def __str__(self):
        """Printing
        
        """
        return f'vertices: {self.vertices}'
    

    def transform(self, R, t):
        """Transform box by rotation R and translation t

        Parameters
        ----------
        R : np.array (2 x 2)
            Rotation matrix
        t : np.array (1 x 2)
            Translation vector
        
        """
        self.min = (R @ self.min[:,None]).T[0] + t 
        self.max = (R @ self.max[:,None]).T[0] + t 


    def halfplanes(self):
        """Compute the halfplane representation of this rectangle

        Returns
        -------
        A : np.array (4 x 2)
            Halfplane vectors
        b : np.array (4 x 1)
            Halfplane scalars

        """
        V = self.vertices
        d = np.diff(V, axis=0, append=V[0][None,:])
        A = np.roll(d, -1, axis=0)  # TODO: normalize A?
        b = block_diag(*list(A)) @ np.reshape(V, (8,1))
        return A, b


    def is_intersecting(self, rect):
        """Check if this rectangle intersects with another rectangle

        Parameters
        ----------
        rect : Rectangle

        Returns
        -------
        bool : True if rectangle intersects with other rectangle
        
        """
        # Check if any of the other rectangle's vertices lie within this rectangle
        A, b = self.halfplanes()
        check = A @ rect.vertices.T >= b
        check = np.all(check, axis=0)
        return np.any(check)
    

    def plot(self, ax=None, color='b'):
        """Plot

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on, if not provided, will generate new set of axes
        color : optional
            Color to plot, default blue
        
        """
        if ax == None:
            fig, ax = plt.subplots()
        
        V = self.vertices
        ax.plot(np.hstack((V[:,0],V[0,0])), np.hstack((V[:,1],V[0,1])), color=color)

