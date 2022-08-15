"""Box class and utilities

This module defines the Box class and relevant utilities.

"""

import numpy as np
import matplotlib.pyplot as plt


class Box:
    """Box class.

    This class represents a 2D axis-aligned box

    Attributes
    ----------
    min : np.array (1 x 2)
        (x_min, y_min) coordinate (lower left) 
    max : np.array (1 x 2)
        (x_max, y_max) coordinate (upper right)
    
    Methods
    -------
    plot()

    """

    def __init__(self, min, max):
        """Constructor
        
        Parameters
        ----------
        min : np.array (1 x 2)
            (x_min, y_min) coordinate (lower left) 
        max : np.array (1 x 2)
            (x_max, y_max) coordinate (upper right)

        """
        if np.all(min < max):
            self.min = min
            self.max = max
        else:
            print("Box is empty")
        

    def __str__(self):
        """Printing
        
        """
        return f'min: {self.min}, max: {self.max}'
    

    def vertices(self):
        """Vertices of box 
        
        Ordered counterclockwise starting with lower left.
        
        Returns
        -------
        np.array (4 x 2)
            Box vertices

        """
        return np.array([self.min, 
                        [self.max[0], self.min[1]],
                         self.max,
                        [self.min[0], self.max[1]]])

    
    def area(self):
        """Area of box

        Returns
        -------
        float
            Box area

        """
        return np.prod(self.max - self.min)
    

    def intersect(self, box):
        """Compute intersection with another box
        
        Parameters
        ----------
        box : Box
            Box to intersect with

        Returns
        -------
        box 
            Box from intersection

        """
        int_min = np.amax(np.vstack((self.min, box.min)), axis=0)
        int_max = np.amin(np.vstack((self.max, box.max)), axis=0)
        if np.all(int_min < int_max):
            return Box(int_min, int_max)
        else:
            #print("Intersection is empty")
            return None
        


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
        
        V = self.vertices()
        ax.plot(np.hstack((V[:,0],V[0,0])), np.hstack((V[:,1],V[0,1])), color=color)
    




def box_from_pts(pts):
    """Extract 2D bounding box set of 2D points

    Parameters
    ----------
    pts : np.array (n_pts x 2)
        Points

    Returns
    -------
    Box
        2D bounding box
        
    """
    min = np.amin(pts, axis=0)
    max = np.amax(pts, axis=0)
    return Box(min, max)