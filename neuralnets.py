"""
Back-propagation neural network implementation, using numpy

Adapted from http://stackoverflow.com/a/3143318
"""

from numpy import *

class NN(object):

  def __init__(self, n_in, n_out, *hidden):
    # Initialize size of each layer
    self.layers = array((n_in,) + hidden + (n_out,))
  
    # Initialize activation values for each node
    # Could use numpy array, but they're updated sequentially anyway
    self.activations = [ones(length) for length in self.layers]
    
    # Initialize weights between levels
    self.weights = [random.uniform(-0.5, 0.5, (i,j)) for i,j in zip(self.layers[:-1], self.layers[1:])]

  def activate(self):
    """Activate all nodes, and output last layer"""
    return random.uniform(-0.5, 0.5, n_out)

  def backPropagate(self, targets, N, M):
    """Bad style bro"""  
    pass

  def train(self, data, targets):
    """Trains the neural network. HAHA rite"""
    pass
