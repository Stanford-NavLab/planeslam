"""PoseGraph class and utilities

This module defines the PoseGraph class and relevant utilities.

"""

import numpy as np
from graphslam.graph import Graph
from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se3 import PoseSE3

from planeslam.geometry.util import R_to_quat, quat_to_R
from planeslam.general import SuppressPrint


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

        Initialize an empty PoseGraph

        """
        self.graph = Graph([], [])
        self.N_vertices = 0
        

    def add_vertex(self, id, pose):
        """Add vertex with ID and pose

        Parameters
        ----------
        id : int
            Vertex ID
        pose : tuple (R,t)
            Tuple of rotation and translation
        
        """
        p = PoseSE3(pose[1], R_to_quat(pose[0]))
        v = Vertex(id, p)
        self.graph._vertices.append(v)
        self.N_vertices += 1

    
    def add_edge(self, ids, transformation, information=np.eye(6)):
        """Add vertex with ID and pose

        Parameters
        ----------
        ids : list
            Pair of vertex IDs for this edge
        transformation : tuple (R,t)
            Tuple of rotation and translation
        
        """
        estimate = PoseSE3(transformation[1], R_to_quat(transformation[0]))
        e = EdgeOdometry(ids, information, estimate)
        self.graph._edges.append(e)
        

    def get_positions(self):
        """Retrieve positions from graph vertices

        Returns
        -------
        positions : np.array (N x 3) 
            Array of positions

        """
        positions = np.zeros((self.N_vertices, 3))
        for i, v in enumerate(self.graph._vertices):
            positions[i] = v.pose.position
        return positions

    
    def get_rotations(self):
        """Retrieve rotations from graph vertices

        Returns
        -------
        rotations : np.array (3 x 3 x N)
            Array of rotation matrices

        """
        rotations = np.zeros((3, 3, self.N_vertices))
        for i, v in enumerate(self.graph._vertices):
            rotations[:,:,i] = quat_to_R(v.pose.orientation)
        return rotations


    def optimize(self):
        """Optimize the pose graph

        """
        # Link edges before calling optimize
        self.graph._link_edges()
        # Suppress output
        with SuppressPrint():
            self.graph.optimize()


    def detect_loop_closures(self):
        """Search for loop closures

        """
        


    def trim_loop_closures(self):
        """
        """
    

    