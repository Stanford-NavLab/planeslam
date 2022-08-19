"""Box3D class and utilities

This module defines the Box3D class and relevant utilities.

"""

import numpy as np
import itertools
import plotly.graph_objects as go


class Box3D:
    """Box3D class.

    This class represents a 3D axis-aligned box

    Vertices are ordered:
      0: - - -
      1: - - +
      2: - + -
      3: - + +
      4: + - -
      5: + - +
      6: + + -
      7: + + +

    Attributes
    ----------
    c : np.array (1 x 3)
        Box center
    G : np.array (3 x 3)
        Box generator matrix
    V : np.array (8 x 3)
        Box vertices
    # min : np.array (1 x 3)
    #     (x_min, y_min, z_min) coordinate 
    # max : np.array (1 x 3)
    #     (x_max, y_max, z_max) coordinate 
    
    Methods
    -------
    plot()

    """

    def __init__(self, c, G):
        """Constructor
        
        Parameters
        ----------
        c : np.array (1 x 3)
            Box center
        G : np.array (3 x 3)
            Box generator matrix

        """
        self.c = c
        self.G = G

        gen_mat = np.asarray(list(itertools.product([-1, 1], repeat=3)))
        self.V = c + gen_mat @ G
        

    def __str__(self):
        """Printing
        
        """
        return f'c: {self.c}, G: {self.G}'
    

    def edges(self):
        """Edges of box 
        
        Returns
        -------
        list
            List of np.array (2 x 3) representing edges of box

        """
        return [np.vstack((self.V[0], self.V[4])),
                np.vstack((self.V[4], self.V[6])),
                np.vstack((self.V[6], self.V[2])),
                np.vstack((self.V[2], self.V[0])),
                np.vstack((self.V[0], self.V[1])),
                np.vstack((self.V[4], self.V[5])),
                np.vstack((self.V[6], self.V[7])),
                np.vstack((self.V[2], self.V[3])),
                np.vstack((self.V[1], self.V[5])),
                np.vstack((self.V[5], self.V[7])),
                np.vstack((self.V[7], self.V[3])),
                np.vstack((self.V[3], self.V[1]))]

    
    def translate(self, t):
        """Return a translated copy of this box
        
        Parameters
        ----------
        t : np.array (1 x 3)
            Translation to apply

        Returns
        -------
        Box3D
            Translated box
        
        """
        return Box3D(self.c + t, self.G)


    def plot_trace(self, color='blue'):
        """Generate plot trace for plotly

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on, if not provided, will generate new set of axes
        color : optional
            Color to plot, default blue
        
        """
        box_data = []
        for e in self.edges():
            box_data.append(go.Scatter3d(x=e[:,0], y=e[:,1], z=e[:,2], marker=dict(color=color), showlegend=False))
        return box_data
        
