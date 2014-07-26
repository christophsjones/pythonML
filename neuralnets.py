"""
Back-propagation neural network implementation, using numpy

Adapted from http://stackoverflow.com/a/3143318
"""

from numpy import *

class NN(object):

  def __init__(self, n_in, n_out, *hidden):
    # Initialize size of each layer
    # One extra input for bias
    self.layers = array((n_in + 1,) + hidden + (n_out,))
    self.n_layers = len(self.layers)

    # Initialize activation values for each node
    # Could use numpy array, but they're updated sequentially anyway
    self.activations = [ones(node_count) for node_count in self.layers]
    
    # Initialize weights between levels
    self.weights = [random.uniform(-0.5, 0.5, (i,j)) for i,j in zip(self.layers[:-1], self.layers[1:])]

    # Initialize previous change in weights, for momentum
    self.change_weights = [zeros([i,j]) for i,j in zip(self.layers[:-1], self.layers[1:])]

  def activate(self, input):
    """Activate all nodes, and output last layer"""
    self.activations[0] = input
    for i in xrange(self.n_layers - 1):
      self.activations[i + 1] = self.weights.dot(self.activations[i])

    return random.uniform(-0.5, 0.5, n_out)

  def backPropagate(self, targets, N, M):
    """Bad style bro"""  
    pass

  def train(self, data, targets):
    """Trains the neural network"""
    for i in range(num_epochs):
      self.backPropagate(targets, 0.1, 0.01)
