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


    def edges(self):
        """Return edges of rectangle

        Ordered CCW starting from (v1->v2)
        
        Returns
        -------
        np.array (4 x 2)
            Edge of rectangle in rows
        
        """
        V = self.vertices
        d = np.diff(V, axis=0, append=V[0][None,:])
        return np.roll(d, -1, axis=0)  


    def halfplanes(self):
        """Compute the halfplane representation of this rectangle

        Matrix A and vector b such that {x | A*x >= b} represents the rectangle

        Returns
        -------
        A : np.array (4 x 2)
            Halfplane vectors
        b : np.array (4 x 1)
            Halfplane scalars

        """
        V = self.vertices
        A = self.edges() / np.linalg.norm(self.edges(), axis=1)[:,None]
        b = block_diag(*list(A)) @ np.reshape(V, (8,1))
        return A, b


    def is_intersecting(self, rect):
        """Check if this rectangle intersects with another rectangle

        Use separating axis test: check if there is a line which separates the
        two rectangles.

        Parameters
        ----------
        rect : Rectangle

        Returns
        -------
        bool : True if rectangle intersects with other rectangle
        
        """
        # Check which side of this rectangles edges the other rectangle's vertices lie on and vice versa
        A, b = self.halfplanes()
        check_self = A @ rect.vertices.T <= b
        check_self = np.all(check_self, axis=1)
        check_self = np.any(check_self)  # are any sides of self rectangle a separating axis

        A, b = rect.halfplanes()
        check_other = A @ self.vertices.T <= b
        check_other = np.all(check_other, axis=1)
        check_other = np.any(check_other) # are any sides of other rectangle a separating axis

        # If there is a separating axis, then rectangles are not intersecting
        return not (check_self | check_other) 
    

    def contains(self, rect):
        """Check if a rectangle is contained in this rectangle

        Parameters
        ----------
        rect : Rectangle
            Point

        Returns
        -------
        bool : True if rectangle is contained, false otherwise
        
        """
        A, b = self.halfplanes()
        return np.all(A @ rect.vertices.T >= b)


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

