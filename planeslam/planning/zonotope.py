"""Zonotope class

This module defines the Zonotope class.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog


class Zonotope(object):
    """ Zonotope class

    TODO: 
      - handle no generators case
      - delete_zeros function
          

    Attributes
    ----------
    dim : int
        Dimension (denoted as n)
    order : int
        Number of generators (denoted as m)
    c : np.array (n x 1)
        Center
    G : np.array (n x m)
        Generators
    Z : np.array (n x m+1)
        Matrix form [c, G]

    Methods
    -------


    Example usage:
        z = Zonotope(np.zeros((2,1)),np.eye(2))

    """
    __array_priority__ = 1000  # Prioritize class mul over numpy array mul

    def __init__(self, center, generators):
        """ Constructor"""
        self.c = center 
        self.G = generators
        self.Z = np.hstack((center, generators))
        self.dim = center.shape[0]
        self.order = generators.shape[1]


    ### ====== Printing ====== ###
    def __str__(self):
        #return "center:\n {0} \n generators:\n {1}".format(self.c, self.G)
        np.set_printoptions(precision=3)
        ind = '\t'
        c_str = ind + str(self.c).replace('\n','\n' + ind)
        G_str = ind + str(self.G).replace('\n','\n' + ind)
        print_str = 'Center:\n' + c_str + '\nGenerators:\n' + G_str 
        return print_str


    ### ====== Operations ====== ###
    def __add__(self, other):
        """ Minkowski addition (overloads '+') """
        # Other is a vector
        if type(other) == np.ndarray:
            c = self.c + other
            G = self.G 
        # Other is a zonotope
        else:
            c = self.c + other.c
            G = np.hstack((self.G, other.G))
        return Zonotope(c,G)


    def __rmul__(self, other):
        """ Right linear map (overloads '*') """
        # Other is a scalar
        if np.isscalar(other):
            c = other * self.c
            G = other * self.G 
        # Other is a matrix
        elif type(other) is np.ndarray:
            c = other @ self.c
            G = other @ self.G 
        return Zonotope(c,G) 
    

    def __mul__(self, other):
        """ (Left) linear map (overloads '*') """
        # Other is a scalar
        if np.isscalar(other):
            c = other * self.c
            G = other * self.G 
        # Other is a matrix
        elif type(other) is np.ndarray:
            c = self.c @ other
            G = self.G @ other
        return Zonotope(c,G) 


    def sample(self, n_points):
        """Sample
        
        (Uniformly) randomly sample points from the interior of the zonotope

        Parameters
        ----------
        n_points : int
            Number of points to sample

        Returns
        -------
        P : np.array (dim x n_points)
            Sampled points

        """
        c = self.c
        G = self.G 
        factors = -1 + 2 * np.random.rand((G.shape[1],n_points))
        p = c + G @ factors
        return p

    
    def augment(self, Z):
        """Augment with another zonotope

        Stacks center and generators together to form new zonotope
        
        Parameters
        ----------
        Z : Zonotope
            Zonotope to augment with (must have same order)

        Returns
        -------
        Zonotope
            Augmented zonotope

        """
        c = np.vstack((self.c, Z.c))
        G = np.vstack((self.G, Z.G))
        return Zonotope(c,G)

    
    def slice(self, dim, slice_pt):
        """Slice zonotope along dim 
        
        Parameters
        ----------
        dim : int or tuple
            Dimension(s) to slice
        slice_pt : 
        
        Returns
        -------
        Zonotope
            Sliced zonotope

        """
        c = self.c; G = self.G
        # TODO: finish implementing this


    def halfspace(self):
        """Generate halfspace representation
        
        Returns
        -------
        A : np.array () 
        b : np.array ()

        """


    def contains(self, x):
        """Check if point x is contained in zonotope.
        
        Parameters
        ----------
        x : np.array (dim x 1)
            Point to check for containment

        Returns
        -------
        bool
            True if x in zonotope, False if not.

        """
        # TODO: finish implementation



    ### ====== Properties ====== ### 
    def vertices(self):
        """ Vertices of zonotope 
        
        Adapted from CORA \@zonotope\vertices.m and \@zonotope\polygon.m
        Tested on 2D zonotopes (n==2)

        Returns
        -------
        V : np.array
            Vertices 

        """
        # Extract variables
        c = self.c
        G = self.G
        n = self.dim
        m = self.order

        if n == 1:
            # Compute the two vertices for 1-dimensional case
            temp = np.sum(np.abs(self.G))
            V = np.array([self.c - temp, self.c + temp])
        elif n == 2:
            # Obtain size of enclosing intervalhull of first two dimensions
            xmax = np.sum(np.abs(G[0,:]))
            ymax = np.sum(np.abs(G[1,:]))

            # Z with normalized direction: all generators pointing "up"
            Gnorm = G
            Gnorm[:,G[1,:]<0] = Gnorm[:,G[1,:]<0] * -1

            # Compute angles
            angles = np.arctan2(G[1,:],G[0,:])
            angles[angles<0] = angles[angles<0] + 2 * np.pi

            # Sort all generators by their angle
            IX = np.argsort(angles)

            # Cumsum the generators in order of angle
            V = np.zeros((2,m+1))
            for i in range(m):
                V[:,i+1] = V[:,i] + 2 * Gnorm[:,IX[i]] 

            V[0,:] = V[0,:] + xmax - np.max(V[0,:])
            V[1,:] = V[1,:] - ymax 

            # Flip/mirror upper half to get lower half of zonotope (point symmetry)
            V = np.block([[V[0,:], V[0,-1] + V[0,0] - V[0,1:]],
                          [V[1,:], V[1,-1] + V[1,0] - V[1,1:]]])

            # Consider center
            V[0,:] = c[0] + V[0,:]
            V[1,:] = c[1] + V[1,:]

        else:
            #TODO: delete aligned and all-zero generators

            # Check if zonotope is full-dimensional
            if self.order < n:
                #TODO: verticesIterateSVG
                print("Vertices for non full-dimensional zonotope not implemented yet - returning empty array")
                V = np.empty()
                return V
            
            # Generate vertices for a unit parallelotope
            vert = np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).reshape(3,-1)
            V = c + G[:,:n] @ vert 
            
            #TODO: rest unimplemented

        return V
            

    ### ====== Plotting ====== ###
    def plot(self, ax=None, color='b', alpha=0.5):
        """Plot function 
        
        Parameters 
        ----------
        ax : matplotlib.axes
            Axes to plot on, if unspecified, will generate and plot on new set of axes
        color : color 
            Plot color
        alpha : float (from 0 to 1)
            Patch transparency

        """
        V = self.vertices()
        xmin = np.min(V[0,:]); xmax = np.max(V[0,:])
        ymin = np.min(V[1,:]); ymax = np.max(V[1,:])

        if ax == None:
            fig, ax = plt.subplots()
        poly = Polygon(V.T, True, color=color, alpha=alpha)
        ax.add_patch(poly)

        # Recompute the ax.dataLim
        ax.relim()
        # Update ax.viewLim using the new dataLim
        ax.autoscale_view()


def is_empty_con_zonotope(A, b):
    """Check if constrained zonotope is empty. 
    
    Used to detect intersection between reach set and unsafe set.
    Implemented by Adam Dai, Derek Knowles (http://cs229.stanford.edu/proj2021spr/report2/81976691.pdf)
    """

    # Dimension of problem
    d = A.shape[1]

    # Cost
    f_cost = np.zeros((d, 1))
    f_cost = np.concatenate((f_cost, np.eye(1)), axis=0)

    # Inequality cons
    A_ineq = np.concatenate((-np.eye(d), -np.ones((d, 1))), axis=1)
    A_ineq = np.concatenate((A_ineq, np.concatenate((np.eye(d), -np.ones((d, 1))), axis=1)), axis=0)
    b_ineq = np.zeros((2 * d, 1))

    # Equality cons
    A_eq = np.concatenate((A, np.zeros((A.shape[0], 1))), axis=1)
    b_eq = b

    res = linprog(f_cost, A_ineq, b_ineq, A_eq, b_eq, (None, None))
    x = res.x
    
    if x[-1] <= 1:
        return False
    return True