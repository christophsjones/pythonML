"""
Back-propagation neural network, using numpy

Heavily modified from http://stackoverflow.com/a/3143318
"""

from numpy import *

# Sigmoid function used in this neural network
def sigmoid(xs):
  """Compute sigmoid at each x in vector"""
  return 2.0 / (1.0 + exp(-xs)) - 1.0

def dsigmoid(ys):
  """Compute sigmoid derivative from each y in vector"""
  return 0.5 * (1.0 - ys*ys)

class NN(object):

  def __init__(self, n_in, *hidden):
    hidden, n_out = hidden[:-1], hidden[-1]
    # Initialize size of each layer
    # One extra input for biases
    self.layers = array((n_in,) + hidden + (n_out,))
    self.n_layers = len(self.layers)

    # Initialize activation values for each node
    # Could use numpy array, but they're updated sequentially anyway
    self.activations = [ones(node_count) for node_count in self.layers]
    
    # Initialize weights between levels, and past changes for momentum
    self.weight_shapes = zip(self.layers[1:], self.layers[:-1] + 1)
    
    self.weights = [random.uniform(-0.5, 0.5, shape) for shape in self.weight_shapes]
    self.past_change = [zeros(shape) for shape in self.weight_shapes]

  def activate(self, inputs):
    """Activate all nodes, and output last layer"""
    self.activations[0] = inputs

    # Append bias term, and don't use sigmoid on input
    self.activations[1] = self.weights[0].dot(append(self.activations[0], 1.0))

    # Activate each hidden layer, using sigmoid function
    for i in xrange(1, self.n_layers - 1):
      self.activations[i + 1] = self.weights[i].dot(append(sigmoid(self.activations[i]), 1.0))

    return self.activations[-1][:]

  def backPropagate(self, targets, a, M):
    """Update weights assuming neural network is activated to achieve targets""" 

    current_change = [zeros(shape) for shape in self.weight_shapes]
    
    error = self.activations[-1] - targets
    # Return square error
    err = dot(error, error) 
    deltas = error


    for i in reversed(xrange(1, self.n_layers - 1)):
      current_change[i] = outer(deltas, append(sigmoid(self.activations[i]), 1.0))
      
      error = self.weights[i].T.dot(deltas)
      deltas = dsigmoid(self.activations[i]) * error[:-1]
    
    current_change[0] = outer(deltas, append(self.activations[0], 1.0))

    for i in xrange(self.n_layers - 1):
      self.weights[i] -= M*self.past_change[i] + a*current_change[i]
      self.past_change[i] = current_change[i]

    return err

  def train(self, data, targets, num_epochs=1000, a=0.02, M=0.002, e=0.000001, verbose=True):
    """Trains the neural network"""

    past_error = -2 * e
    error = 0.0
    for i in xrange(num_epochs):
      error = 0.0
      for x,y in zip(data, targets):
        self.activate(x)
        error += self.backPropagate(y, a, M)
      if verbose and i % max(num_epochs / 10, 1) == 0:
        print "Iteration: %s of %s, Error: %s" % (i,num_epochs, (error / len(data)) ** 0.5)
      
      if abs(past_error - error) <= 0:
        if verbose:
          print "Converged on iteration %s of %s, Error: %s" % (i, num_epochs, (error / len(data)) ** 0.5)
        break
      past_error = error

    # Return RMS error
    return (error / len(data)) ** 0.5