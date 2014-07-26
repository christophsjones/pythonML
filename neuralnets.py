"""
Back-propagation neural network implementation, using numpy

Adapted from http://stackoverflow.com/a/3143318
"""

from numpy import *

class NN(object):

  def __init__(self, n_in, n_out, *hidden):
    # Initialize size of each layer
    self.layers = array((n_in,) + hidden + (n_out,))
    
