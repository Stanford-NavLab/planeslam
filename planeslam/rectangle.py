"""Rectangle class and utilities

This module defines the Rectangle class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt


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
            Rectangle vertices

        """
        if np.all(min < max):
            self.min = min
            self.max = max
        else:
            print("Box is empty")
        

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

