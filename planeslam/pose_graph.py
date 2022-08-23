"""PoseGraph class and utilities

This module defines the PoseGraph class and relevant utilities.

"""

import numpy as np
from graphslam.graph import Graph
from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se3 import PoseSE3


class PoseGraph:
    """PoseGraph class.

    This class is a wrapper around the graphslam Graph class. 

    Attributes
    ----------
    graph : Graph
        graphslam Graph
    
    
    Methods
    -------
    plot()

    """

    def __init__(self):
        """Constructor
        
        Parameters
        ----------
        

        """
        
        

    def add_vertex(self, id, pose):
        """Add vertex with ID and pose

        Parameters
        ----------
        id : int
            Vertex ID
        pose : tuple (R,t)
            Tuple of rotation and translation
        
        """
    
    def add_edge(self, ids, transformation):
        """Add vertex with ID and pose

        Parameters
        ----------
        ids : list
            Pair of vertex IDs for this edge
        transformation : tuple (R,t)
            Tuple of rotation and translation
        
        """


    def get_positions(self):
        """
        """

    
    def get_rotations(self):
        """
        """


    def optimize(self):
        """
        """

    def detect_loop_closures(self):
        """
        """

    def trim_loop_closures(self):
        """
        """
    

    